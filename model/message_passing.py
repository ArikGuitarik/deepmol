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
        :return: generated convolution filters. If hparams.use_matrix_filters is True, the shape is
            [batch_size, num_atoms, num_atoms, hidden_state_dim, hidden_state_dim],
            else we have a vector for each atom pair: [batch_size, num_atoms, num_atoms, hidden_state_dim]
        """
        batch_size = tf.shape(distance_matrix)[0]
        num_atoms = tf.shape(distance_matrix)[1]
        hidden_state_dim = self.hparams.hidden_state_dim
        use_matrix_filters = self.hparams.use_matrix_filters

        # map every distance value in the whole batch separately:
        dist_mat_reshaped = tf.reshape(distance_matrix, [batch_size, num_atoms, num_atoms, 1], name="dist_mat_reshaped")

        layer_dims = np.ones(self.hparams.filter_hidden_layers + 1) * self.hparams.filter_hidden_dim
        layer_dims[-1] = hidden_state_dim ** 2 if use_matrix_filters else hidden_state_dim  # output dim
        activation = tf.nn.leaky_relu if self.hparams.use_leaky_relu else tf.nn.relu
        fc_nn = FullyConnectedNN(self.hparams, layer_dims, activation, output_activation=None)
        conv_filter = fc_nn.forward(dist_mat_reshaped)

        if use_matrix_filters:
            conv_filter = tf.reshape(conv_filter,
                                     [batch_size, num_atoms, num_atoms, hidden_state_dim, hidden_state_dim],
                                     name='matrix_filters')
        else:
            conv_filter = tf.reshape(conv_filter, [batch_size, num_atoms, num_atoms, hidden_state_dim],
                                     name='vector_filters')

        return conv_filter


class MatrixMessagePassing(Model):
    """Implements EdgeNetwork message function from MPNN paper.

    To generate the message from atom j to atom i, the filter matrix belonging to the distance between i and j
    is multiplied with the hidden state of j. All messages to atom i are summed and a bias is added.

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    """

    def __init__(self, hparams):
        super(MatrixMessagePassing, self).__init__(hparams)

    def _forward(self, hidden_states, matrix_filters):
        """Forward pass for generating messages using matrix filters.

        :param hidden_states: Hidden states of all atoms, shaped [batch_size, num_atoms, hidden_state_dim]
        :param matrix_filters: generated convolution filters, shaped
            [batch_size, num_atoms, num_atoms, hidden_state_dim, hidden_state_dim]
        :return: sum of incoming messages to each atom, shaped [batch_size, num_atoms, hidden_state_dim]
        """
        batch_size = tf.shape(hidden_states)[0]
        num_atoms = tf.shape(hidden_states)[1]
        hidden_state_dim = self.hparams.hidden_state_dim

        # multiply matrix with hidden states and add bias to generate messages
        hidden_states_flat = tf.reshape(hidden_states, [batch_size, num_atoms * hidden_state_dim, 1],
                                        name='hidden_states_flat')
        matrix_filters = tf.transpose(matrix_filters, [0, 1, 3, 2, 4])
        matrix_filters = tf.reshape(matrix_filters,
                                    [batch_size, num_atoms * hidden_state_dim, num_atoms * hidden_state_dim],
                                    name='matrix_filters_flat')
        messages = tf.matmul(matrix_filters, hidden_states_flat)  # (b, n*d, n*d) x (b, n*d, 1) -> (b, n*d, 1)
        messages = tf.reshape(messages, [batch_size * num_atoms, hidden_state_dim], name='messages_unbiased')
        messages += tf.get_variable("message_bias", shape=hidden_state_dim)
        messages = tf.reshape(messages, [batch_size, num_atoms, hidden_state_dim], name='messages')

        return messages

# todo add vector message passing
