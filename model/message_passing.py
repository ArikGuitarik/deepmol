import tensorflow as tf
from .abstract_model import Model
import numpy as np
from .fc_nn import FullyConnectedNN


class ConvFilterGenerator(Model):
    """A model mapping each interatomic distance value to a matrix or vector as the basis for message generation.

    Message generation works similar to regular convolutions for images with a discrete pixel grid.
    Here however, distance values are continuous which makes such a filter generator necessary.
    Depending on the method (the continuous convolutions in SchNet or EdgeNet in the MPNN paper), either a vector
    or a matrix are output for each distance value.

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    """

    def __init__(self, hparams):
        super(ConvFilterGenerator, self).__init__(hparams)

    def _forward(self, distance_matrix):
        """Forward pass of filter generator.

        :param distance_matrix: distance matrix shaped [batch_size, num_atoms, num_atoms]
        :return: generated convolution filters. If hparams.use_matrix_filter is True, the shape is
            [batch_size, num_atoms, num_atoms, hidden_state_dim, hidden_state_dim],
            else we have a vector for each atom pair: [batch_size, num_atoms, num_atoms, hidden_state_dim]
        """
        batch_size = tf.shape(distance_matrix)[0]
        num_atoms = tf.shape(distance_matrix)[1]
        hidden_state_dim = self.hparams.hidden_state_dim
        use_matrix_filter = self.hparams.use_matrix_filter

        # map every distance value in the whole batch separately:
        dist_mat_reshaped = tf.reshape(distance_matrix, [batch_size, num_atoms, num_atoms, 1], name="dist_mat_reshaped")

        layer_dims = np.ones(self.hparams.filter_hidden_layers + 1) * self.hparams.filter_hidden_dim
        layer_dims[-1] = hidden_state_dim ** 2 if use_matrix_filter else hidden_state_dim  # output dim
        activation = tf.nn.leaky_relu if self.hparams.use_leaky_relu else tf.nn.relu
        fc_nn = FullyConnectedNN(self.hparams, layer_dims, activation, output_activation=None)
        conv_filter = fc_nn.forward(dist_mat_reshaped)

        if use_matrix_filter:
            conv_filter = tf.reshape(conv_filter,
                                     [batch_size, num_atoms, num_atoms, hidden_state_dim, hidden_state_dim],
                                     name='filter_matrix')
        else:
            conv_filter = tf.reshape(conv_filter, [batch_size, num_atoms, num_atoms, hidden_state_dim],
                                     name='filter_vector')

        return conv_filter

# todo: add message passing methods below here:
