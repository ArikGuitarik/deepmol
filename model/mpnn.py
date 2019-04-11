import tensorflow as tf
import numpy as np
from .message_passing import MatrixMessagePassing, VectorMessagePassing, ConvFilterGenerator
from .update_function import GRUUpdate
from .read_out import Set2Vec
from .fc_nn import FullyConnectedNN
from .abstract_model import Model


class MPNN(Model):
    """Message Passing Neural Network.

    Creates a vector embedding of the entire molecule and maps it to the molecule properties.
    Behaves mostly like the model described by Gilmer et al.

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    :param output_dim: output dimension of the network. for prediction, the number of properties to be learned at once.
        If None, the output network will be omitted and the vector embedding of the molecule is returned.
    """

    def __init__(self, hparams, output_dim=None):
        super(MPNN, self).__init__(hparams)
        self.output_dim = output_dim

        self.filter_gen = ConvFilterGenerator(hparams)
        self.read_out = Set2Vec(hparams)

        # If weight tying is enabled, the message passing and update models are re-used throughout the forward pass.
        if hparams.weight_tying:
            self._message_passing = MatrixMessagePassing(
                hparams) if hparams.use_matrix_filters else VectorMessagePassing(hparams)
            self._update = GRUUpdate(hparams)

    @property
    def message_passing(self):
        """Return the message passing model. If weight tying is disabled, create a new one, else reuse."""
        if self.hparams.weight_tying:
            return self._message_passing
        else:
            return MatrixMessagePassing(self.hparams) if self.hparams.use_matrix_filters else VectorMessagePassing(
                self.hparams)

    @property
    def update(self):
        """Return the update function model. If weight tying is disabled, create a new one, else reuse."""
        if self.hparams.weight_tying:
            return self._update
        else:
            return GRUUpdate(self.hparams)

    def _forward(self, molecules):
        """Forward pass of the message passing neural network.

        :param molecules: batch of molecules as a TFMolBatch objects
        :return: tensor of predicted properties or molecule embedding vector, as specified by output_dim.
        """
        # zero-pad up to hidden state dimension
        num_atom_features = molecules.atoms.get_shape()[2].value
        with tf.control_dependencies([tf.assert_less_equal(num_atom_features, self.hparams.hidden_state_dim)]):
            hidden_states = tf.pad(molecules.atoms,
                                   [[0, 0], [0, 0], [0, self.hparams.hidden_state_dim - num_atom_features]])

        filters = self.filter_gen.forward(molecules.distance_matrix)
        # perform message passing
        for i in range(self.hparams.num_propagation_steps):
            messages = self.message_passing.forward(hidden_states, filters)
            hidden_states = self.update.forward(hidden_states, messages, molecules.mask)

        # read-out
        features_and_states = [molecules.atoms, hidden_states]
        features_and_states = tf.concat(features_and_states, axis=2, name="atoms_concat")
        mol_embedding = self.read_out.forward(features_and_states, mask=molecules.mask)

        if self.output_dim is None:
            return mol_embedding

        # map vector embedding of the entire molecule to the output
        layer_dims = np.ones(self.hparams.mpnn_out_hidden_layers + 1) * self.hparams.mpnn_out_hidden_dim
        layer_dims[-1] = self.output_dim  # output dim
        activation = tf.nn.leaky_relu if self.hparams.use_leaky_relu else tf.nn.relu
        fc_nn = FullyConnectedNN(self.hparams, layer_dims=layer_dims, activation=activation, output_activation=None)
        output = fc_nn.forward(mol_embedding)

        return output

    @staticmethod
    def default_hparams():
        """Return a tf.contrib.training.HParams with the default hyperparameter configuration."""
        return tf.contrib.training.HParams(
            num_propagation_steps=3,
            mpnn_out_hidden_layers=1,
            mpnn_out_hidden_dim=200,
            filter_hidden_layers=4,
            filter_hidden_dim=50,
            use_matrix_filters=True,  # otherwise use vector
            set2vec_steps=12,
            set2vec_num_attention_heads=1,
            hidden_state_dim=50,
            use_leaky_relu=True,
            weight_tying=True
        )
