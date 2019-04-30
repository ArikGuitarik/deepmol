from .mpnn import MPNN
import numpy as np
from scipy.special import comb
import tensorflow as tf
from .fc_nn import FullyConnectedNN
from data.molecules import TFMolBatch


class MolVAE:
    """A molecular geometry-based variational autoencoder.

    Also usable as autoencoder by setting the hyperparameter "variational" to False.
    The encoder is based on the message passing phase of an MPNN.
    The decoder is based on a fully-connected neural network. Atom types and geometry are each predicted by a separate
    output layer on top of multiple shared network layers.

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    :param max_num_atoms: Maximum number of atoms in a molecule. (Smaller molecules are zero-padded.)
    :param num_atom_types: Number of atom types (including "none/padded").
    """

    def __init__(self, hparams, max_num_atoms, num_atom_types):
        self.prior = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(hparams.latent_dim))
        self.encode = tf.make_template('Encoder', self._encode)
        self.decode = tf.make_template('Decoder', self._decode)
        self.reconstruct = tf.make_template('Reconstruct', self._reconstruct)
        self.num_atom_types = num_atom_types
        self.max_num_atoms = max_num_atoms
        self.atom_out_dim = max_num_atoms * num_atom_types
        self.distances_out_dim = int(max_num_atoms * (max_num_atoms - 1) / 2)  # = 1+2+3+...+(num_nodes - 1)
        self.hparams = hparams

    def _encode(self, mols):
        """Encoder, applies a neural network to the concatenation of the hidden atom states after message passing.

        :param mols: batch of molecules as a TFMolBatch object
        :return: multivariate normal distribution in latent space.
        """
        mpnn = MPNN(self.hparams, output_dim=None)
        h_concat = mpnn.forward(mols)

        # output network
        layer_dims = np.ones(self.hparams.encoder_out_hidden_layers) * self.hparams.encoder_out_hidden_dim
        activation = tf.nn.leaky_relu if self.hparams.use_leaky_relu else tf.nn.relu
        fc_nn = FullyConnectedNN(self.hparams, layer_dims, activation=activation, output_activation=activation)
        h_concat = fc_nn.forward(h_concat)
        loc = tf.layers.dense(h_concat, self.hparams.latent_dim, name='latent_mean')
        scale = tf.layers.dense(h_concat, self.hparams.latent_dim, tf.nn.softplus,  # softplus -> positive values
                                name='latent_variance', trainable=self.hparams.variational) + 1e-5
        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)

    def _decode(self, z):
        """ Decode a batch of latent vectors back to molecules.

        :param z: batch of latent vectors of shape [batch_size, latent_dim]
        :return: TFMolBatch of decoded molecules
        """
        # shared layers:
        layer_dims = np.ones(self.hparams.decoder_hidden_layers) * self.hparams.decoder_hidden_dim
        activation = tf.nn.leaky_relu if self.hparams.use_leaky_relu else tf.nn.relu
        fc_nn = FullyConnectedNN(self.hparams, layer_dims, activation=activation, output_activation=activation)
        z = fc_nn.forward(z)

        # output layer for atom reconstruction
        atom_logits = tf.layers.dense(z, self.atom_out_dim)
        batch_size = tf.shape(z)[0]
        atom_matrix_logits = tf.reshape(atom_logits, [batch_size, self.max_num_atoms, self.num_atom_types])

        if self.hparams.coordinate_output:
            coordinates = tf.layers.dense(z, 3 * self.max_num_atoms)
            coordinates = tf.reshape(coordinates, [batch_size, self.max_num_atoms, 3], name='coordinates')
            decoded_mols = TFMolBatch(atoms_logits=atom_matrix_logits, coordinates=coordinates)
        else:
            edges = tf.layers.dense(z, self.distances_out_dim, tf.nn.softplus)  # softplus ensures positivity
            decoded_mols = TFMolBatch(atoms_logits=atom_matrix_logits, distances=edges)

        return decoded_mols

    def calculate_loss(self, mols):
        """Perform forward pass and calculate the loss (negative ELBO) for the given molecule batch.

        If hparams.variational is False, only the reconstruction loss is accounted for.
        If hparams.coordinate_output is False (thus, distances are output):
        To penalize violations of the triangle inequality, an additional geometric penalty term is added, weighted
        by hparams.geometric_penalty_weight.

        :param mols: batch of molecules as a TFMolBatch object
        :return: loss tensor
        """
        posterior = self.encode(mols)
        if self.hparams.variational:
            z = posterior.sample()
        else:
            z = posterior.mean()
        decoded_mols = self.decode(z)
        with tf.name_scope('loss'):
            loss = self.reconstruction_loss(mols, decoded_mols)

            if not self.hparams.coordinate_output:
                geom_penalty = self.geometric_penalty(decoded_mols)
                tf.summary.scalar('geom_penalty', geom_penalty)
                loss += geom_penalty * self.hparams.geometric_penalty_weight

            if self.hparams.variational:
                kl_divergence = tf.reduce_mean(tf.distributions.kl_divergence(posterior, self.prior))
                tf.summary.scalar('kl_divergence', kl_divergence)
                loss += kl_divergence * self.hparams.beta

            return loss

    def _reconstruct(self, mols):
        """Forward pass through the (variational) autoencoder.

        :param mols: batch of molecules as a TFMolBatch object
        :return: TFMolBatch of reconstructed molecules
        """
        posterior = self.encode(mols)
        if self.hparams.variational:
            z = posterior.sample()
        else:
            z = posterior.mean()
        decoded_mols = self.decode(z)

        return decoded_mols

    def _get_triangle_sets_and_weights(self, decoded_mols):
        """Extract triplets for triangle inequality and the respective existence probabilities of all involved atoms.

        To calculate the geometric penalty, the triangle inequality needs to be checked for all sets of three atoms.
        Additionally, the penalty is weighted by the existence probability of the involved atoms since violations by
        zero-padded atoms are irrelevant.

        :param decoded_mols: TFMolBatch of decoded molecules
        :return:
            - triangle_sets: distance values [d_ij, d_ik, d_jk] for all i<j<k, shaped [batch_size, num_sets, 3]
            - weights: joint existence probability of each set, shaped [batch_size, num_sets]
        """
        decoded_mask = 1 - decoded_mols.atoms[:, :, -1]  # existence probability of atom = 1 - p(type=None)

        # compile indices to extract/collect
        num_sets = comb(self.max_num_atoms, 3, exact=True)  # number of sets of three atoms
        weight_indices = np.zeros([num_sets, 3, 1])
        set_indices = np.zeros([num_sets, 3, 2])  # num_sets, elements per set, indices in matrix
        set_counter = 0
        for i in range(self.max_num_atoms):  # iterate over all relevant sets of three atoms
            for j in range(i + 1, self.max_num_atoms):
                for k in range(j + 1, self.max_num_atoms):
                    set_indices[set_counter] = [[i, j], [i, k], [j, k]]
                    weight_indices[set_counter, :] = [[i], [j], [k]]
                    set_counter += 1
        with tf.name_scope('triangle_sets'):
            # add batch dimension
            batch_size = tf.shape(decoded_mols.distances)[0]
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.tile(tf.reshape(batch_indices, [batch_size, 1, 1, 1]), [1, num_sets, 3, 1])

            set_indices = tf.tile(set_indices, [batch_size, 1, 1])
            set_indices = tf.reshape(set_indices, [batch_size, num_sets, 3, 2])
            set_indices = tf.to_int32(set_indices)
            set_indices = tf.concat([batch_indices, set_indices], axis=-1)

            weight_indices = tf.tile(weight_indices, [batch_size, 1, 1])
            weight_indices = tf.reshape(weight_indices, [batch_size, num_sets, 3, 1])
            weight_indices = tf.to_int32(weight_indices)
            weight_indices = tf.concat([batch_indices, weight_indices], axis=-1)

            # collect relevant elements
            triangle_sets = tf.gather_nd(params=decoded_mols.distance_matrix, indices=set_indices)
            weights = tf.gather_nd(params=decoded_mask, indices=weight_indices)
            weights = tf.reduce_prod(weights, axis=-1)  # joint probability

        return triangle_sets, weights

    def geometric_penalty(self, decoded_mols):
        """Calculate a penalty term for violations of the triangle inequality.

        :param decoded_mols: TFMolBatch of decoded molecules
        :return: tensor specifying the geometric penalty, summed over atoms, averaged over the batch
        """
        with tf.name_scope('geom_penalty'):
            sets, weights = self._get_triangle_sets_and_weights(decoded_mols)

            violations = 2 * tf.reduce_max(sets, axis=-1) - tf.reduce_sum(sets, axis=-1)
            violations = tf.maximum(violations, 0)
            weighted_violations = violations * weights
            geom_penalty = tf.reduce_sum(weighted_violations, axis=1)  # sum over atoms in each molecule
            geom_penalty = tf.reduce_mean(geom_penalty)  # average over batch
            return geom_penalty

    def reconstruction_loss(self, mols, decoded_mols):
        """Compute the reconstruction loss between original and decoded molecule.

        Atoms are compared using softmax cross entropy, distances using mean squared error.
        Distances are weighted by the probability that both involved atoms exist in both reconstruction and original.

        :param mols: TFMolBatch of original molecules
        :param decoded_mols: TFMolBatch of decoded molecules
        :return: Reconstruction loss tensor
        """
        with tf.name_scope('rec_loss'):
            atom_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=mols.atoms, logits=decoded_mols.atoms_logits)
            atom_loss = tf.reduce_sum(atom_loss, axis=1)  # sum over atoms in each molecule
            atom_loss = tf.reduce_mean(atom_loss)  # average over batch

            dist_weights = self._generate_dist_weights(mols, decoded_mols)
            reduction = tf.losses.Reduction.MEAN  # sum over batch and atoms and divide by sum of weights
            distance_loss = tf.losses.mean_squared_error(mols.distances, decoded_mols.distances, weights=dist_weights,
                                                         reduction=reduction)
            tf.summary.scalar('atom_loss', atom_loss)
            tf.summary.scalar('mask_sum', tf.reduce_mean(tf.reduce_sum(dist_weights, axis=1)))
            tf.summary.scalar('distance_loss', distance_loss)

            reconstruction_loss = self.hparams.gamma * tf.cast(atom_loss, tf.float32)
            reconstruction_loss += (2 - self.hparams.gamma) * distance_loss

            return reconstruction_loss

    def sample(self, num_samples):
        """Sample molecules from the latent prior.

        :param num_samples: The number of molecules to sample.
        :return: TFMolBatch of sampled molecules
        """
        z_values = self.prior.sample(num_samples)
        mols = self.decode(z_values)
        return mols

    def _generate_dist_weights(self, mols, decoded_mols, extract_upper_triangle=True):
        """Generate weights for all distance values, based on the existence probability of the involved atoms.

        When comparing molecules, it is meaningless to compare distances between atoms that only exist in one of them.
        Therefore, each distance between two atoms is weighted by the existence probability of the involved atoms.

        :param mols: TFMolBatch of original molecules
        :param decoded_mols: TFMolBatch of decoded molecules
        :param extract_upper_triangle: If True, the relevant entries of the distance matrix (upper triangle) are
            extracted and flattened. Else, the weights have the shape of the distance matrix.
        :return: Weights, in the form specified by extract_upper_triangle.
        """
        with tf.name_scope('distance_weights'):
            decoded_dist_weights = tf.reshape(decoded_mols.mask, [-1, self.max_num_atoms, 1]) * tf.reshape(
                decoded_mols.mask, [-1, 1, self.max_num_atoms])
            label_dist_weights = tf.reshape(mols.mask, [-1, self.max_num_atoms, 1])
            label_dist_weights *= tf.reshape(mols.mask, [-1, 1, self.max_num_atoms])
            combined_dist_weights = decoded_dist_weights * tf.cast(label_dist_weights, tf.float32)

            if extract_upper_triangle:
                # create TFMolBatch for the conversion between distance matrix -> distances
                dummy_atoms = mols.atoms  # irrelevant, but required parameter
                dummy_mols = TFMolBatch(atoms=dummy_atoms, distance_matrix=combined_dist_weights)
                flattened_dist_weights = dummy_mols.distances
                return flattened_dist_weights

            return combined_dist_weights

    @staticmethod
    def default_hparams():
        """Return a tf.contrib.training.HParams with the default hyperparameter configuration."""
        hparams = MPNN.default_hparams()
        hparams.set_hparam('learning_rate', 6.25e-4)
        hparams.set_hparam('use_set2vec', False)
        hparams.set_hparam('hidden_state_dim', 110)
        hparams.add_hparam('latent_dim', 64)  # dimension of latent space
        hparams.add_hparam('encoder_out_hidden_dim', 100)  # of the network mapping to hidden states to latent space
        hparams.add_hparam('encoder_out_hidden_layers', 2)  # of the network mapping to hidden states to latent space
        hparams.add_hparam('decoder_hidden_dim', 200)  # for shared layers of decoder network
        hparams.add_hparam('decoder_hidden_layers', 5)  # for shared layers of decoder network
        hparams.add_hparam('variational', True)  # use as VAE as opposed to a regular autoencoder
        hparams.add_hparam('coordinate_output', False)  # output coordinates instead of distances
        hparams.add_hparam('geometric_penalty_weight', 0.0)  # penalize violations of the triangle inequality
        hparams.add_hparam('gamma', 8e-3)  # relative weight on distance vs atom reconstruction
        hparams.add_hparam('beta', 1e-3)  # weight on kl_div term, as in a beta-VAE
        return hparams
