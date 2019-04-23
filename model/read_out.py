import tensorflow as tf
from .abstract_model import Model
from .fc_nn import FullyConnectedNN


class Set2Vec(Model):
    """Implements the Set2Vec read-out, mapping the hidden states to a vector embedding of the entire molecule.

    Additionally, multi-head attention is possible by setting hparams.set2vec_num_attention_heads > 1.
    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    """

    def __init__(self, hparams):
        super(Set2Vec, self).__init__(hparams)

    @staticmethod
    def _lstm(m, c, hidden_state_dim, name=""):
        """Create a special LSTM cell with no inputs.

        Usually, an LSTM cell produces its output based on its previous output, previous cell state and new input.
        In this model however, the LSTM does not receive any input.
        Moreover, only the cell state is directly handed on to the next time step while the output
        is transformed by an attention block before being passed on.

        :param m: the previous output of the cell
        :param c: the previous cell state
        :param hidden_state_dim: dimension of the hidden states
        :param name: prefix for variable names
        :return: new cell state and output
        """
        # Input Gate
        m_dim = 2 * hidden_state_dim
        w_im = tf.get_variable(name + "w_im", [m_dim, hidden_state_dim])
        b_i = tf.get_variable(name + "b_i", [1, hidden_state_dim], initializer=tf.zeros_initializer)
        i_t = tf.sigmoid(m @ w_im + b_i, name="i_t")

        # Forget Gate
        w_fm = tf.get_variable(name + "w_fm", [m_dim, hidden_state_dim])
        b_f = tf.get_variable(name + "b_f", [1, hidden_state_dim], initializer=tf.zeros_initializer)
        f_t = tf.sigmoid(m @ w_fm + b_f, name="f_t")

        # Cell State
        w_cm = tf.get_variable(name + "w_cm", [m_dim, hidden_state_dim])
        b_c = tf.get_variable(name + "b_c", [1, hidden_state_dim], initializer=tf.zeros_initializer)

        # Output Gate
        w_om = tf.get_variable(name + "w_om", [m_dim, hidden_state_dim])
        b_o = tf.get_variable(name + "b_o", [1, hidden_state_dim], initializer=tf.zeros_initializer)
        o_t = tf.sigmoid(m @ w_om + b_o, name="o_t")

        c_new = f_t * c + i_t * tf.tanh(m @ w_cm + b_c)
        m_new = o_t * tf.tanh(c_new)

        return m_new, c_new

    def _forward(self, hidden_states, mask):
        """Forward pass of Set2Vec which maps the hidden states to a vector embedding of the entire molecule.

        :param hidden_states: Hidden states of all atoms, shaped [batch_size, num_atoms, hidden_state_dim]
        :param mask: indicates whether an atom is actually present (1) or zero-padded (0). [batch_size, num_atoms]
        :return: Vector embedding of the entire molecule. [batch_size, 2 * hidden_state_dim]
        :raises ValueError: If the hidden state dimension is not divisible by the number of attention heads.
        """
        # embed hidden states using a neural network as initial step
        hidden_state_dim = hidden_states.get_shape()[2].value
        fc_nn = FullyConnectedNN(self.hparams, layer_dims=[hidden_state_dim], activation=None, output_activation=None)
        hidden_states = fc_nn.forward(hidden_states)

        # prepare multi-head attention
        batch_size = tf.shape(hidden_states)[0]
        num_attention_heads = self.hparams.set2vec_num_attention_heads
        if hidden_state_dim % num_attention_heads != 0:
            raise ValueError(
                'The hidden state dimension (%d) is not divisible by the number of attention heads (%d).' % (
                    hidden_state_dim, num_attention_heads))
        attention_head_dim = int(hidden_state_dim / num_attention_heads)
        hidden_states = tf.reshape(hidden_states, [batch_size, -1, num_attention_heads, attention_head_dim])

        m = tf.zeros([batch_size, 2 * hidden_state_dim])
        c = tf.zeros([batch_size, hidden_state_dim])

        m_to_query = tf.get_variable("m_to_query", [hidden_state_dim, hidden_state_dim])
        attention_v = tf.get_variable("att_v", [num_attention_heads, attention_head_dim])

        large_negative_value = -1e10  # must not be too large either, otherwise NaNs will occur
        mask = tf.cast(mask, tf.float32)
        mask = (1 - mask) * large_negative_value  # masked terms will turn zero in attention softmax
        mask = tf.reshape(mask, [batch_size, -1, 1])

        # variable names in the following comments: n = num_atoms, d = hidden_state_dim, d/k = attention_head_dim
        for i in range(self.hparams.set2vec_steps):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                with tf.name_scope("set2vec_%d" % i):
                    m, c = self._lstm(m, c, hidden_state_dim, name='lstm')
                    query = m @ m_to_query  # [batch, d] @ [d, d] = [batch, d]
                    query = tf.reshape(query, [batch_size, 1, num_attention_heads, attention_head_dim], name='query')
                    # add query to all nodes in batch: [batch, n, k, d/k] + [batch, 1, k, d/k] = [batch, n, k, d/k]
                    tanh_term = tf.tanh(query + hidden_states)
                    # for every node, perform dot product with attention vector v [batch, n, k, d/k] * [k, d/k]
                    energies = tf.reduce_sum(tanh_term * attention_v, axis=3, name='energies')

                    # Apply mask
                    if mask is not None:
                        energies += mask  # [batch_size, n, k] + [batch_size, n, 1]
                    attention = tf.nn.softmax(energies, 1, name='attention')  # [batch_size, n, k]

                    # Attend
                    attention = tf.reshape(attention, [batch_size, -1, num_attention_heads, 1])
                    read = tf.reduce_sum(attention * hidden_states, 1)  # sum_n [batch, n, k, 1] * [batch, n, k, d/k]
                    read = tf.reshape(read, [batch_size, hidden_state_dim])  # reshape back to [batch, d]
                    m = tf.concat([m, read], axis=1, name='m')

        return m


class ConcatReadOut(Model):
    """Simply concatenates the hidden states of all atoms. (Useful if atom order should be preserved.)

    :param hparams: hyperparameters, as a tf.contrib.training.HParams object
    """
    def __init__(self, hparams):
        super().__init__(hparams)

    def _forward(self, hidden_states, mask=None):
        """Concatenate hidden states.

        :param hidden_states: Hidden states of all atoms, shaped [batch_size, num_atoms, hidden_state_dim]
        :param mask: indicates whether an atom is actually present (1) or zero-padded (0). [batch_size, num_atoms]
        :return: concatenated tensor of dimension [batch_size, num_atoms * num_atom_features].
        """
        # set hidden states of all padded atoms to zero
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            hidden_states = hidden_states * mask

        hidden_state_dim = hidden_states.get_shape()[2].value
        batch_size = tf.shape(hidden_states)[0]
        num_atoms = hidden_states.get_shape()[1]
        concatenated_nodes = tf.reshape(hidden_states, [batch_size, num_atoms * hidden_state_dim])

        return concatenated_nodes
