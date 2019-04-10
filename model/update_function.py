import tensorflow as tf
from .abstract_model import Model


class GRUUpdate(Model):
    """Implements the GRU Update function to update the hidden states based on their incoming messages.

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    """

    def __init__(self, hparams):
        super(GRUUpdate, self).__init__(hparams)

    def _forward(self, hidden_states, messages, mask):
        """Forward pass updating each hidden state using its incoming messages.
        In contrast to the original definition, we only use one message per graph edge.

        :param hidden_states: Hidden states of all atoms, shaped [batch_size, num_atoms, hidden_state_dim]
        :param messages: sum of incoming messages for each atom, shaped [batch_size, num_atoms, hidden_state_dim]
        :param mask: indicates whether an atom is actually present (1) or zero-padded (0). [batch_size, num_atoms]
        :return: updated states shaped [batch_size, num_atoms, hidden_state_dim]
        """
        batch_size = tf.shape(hidden_states)[0]
        num_atoms = tf.shape(hidden_states)[1]
        hidden_state_dim = hidden_states.get_shape()[2]

        w_z = tf.get_variable("w_z", shape=[hidden_state_dim, hidden_state_dim])
        u_z = tf.get_variable("u_z", shape=[hidden_state_dim, hidden_state_dim])
        w_r = tf.get_variable("w_r", shape=[hidden_state_dim, hidden_state_dim])
        u_r = tf.get_variable("u_r", shape=[hidden_state_dim, hidden_state_dim])
        w = tf.get_variable("w", shape=[hidden_state_dim, hidden_state_dim])
        u = tf.get_variable("u", shape=[hidden_state_dim, hidden_state_dim])

        # reshape hidden states, messages and mask so that each row = one atom
        hidden_states = tf.reshape(hidden_states, [batch_size * num_atoms, hidden_state_dim])
        messages = tf.reshape(messages, [batch_size * num_atoms, hidden_state_dim])
        mask = tf.cast(tf.reshape(mask, [batch_size * num_atoms, 1]), tf.float32)

        # calculate values for update gate z_t and reset gate r_t
        z_t = tf.sigmoid(messages @ w_z + hidden_states @ u_z, name="z_t")
        r_t = tf.sigmoid(messages @ w_r + hidden_states @ u_r, name="r_t")

        # combine message with previous state
        update = tf.tanh(messages @ w + (r_t * hidden_states) @ u, name="update")

        # update state with update, but keep previous state according to values of update gate z_t
        updated_states = (1 - z_t) * hidden_states + z_t * update

        # zero out masked nodes
        updated_states = updated_states * mask

        # reshape back to original shape
        updated_states = tf.reshape(updated_states, [batch_size, num_atoms, hidden_state_dim], name="updated_states")

        return updated_states
