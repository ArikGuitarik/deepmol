import tensorflow as tf
import numpy as np
from model.read_out import Set2Vec

HParams = tf.contrib.training.HParams


class TestSet2Vec(tf.test.TestCase):
    def test_permutation_invariance(self):
        """Test whether the set2vec output is invariant to permutation of the input mols"""
        num_atoms, batch_size, hidden_state_dim = 4, 3, 5

        hparams = HParams(hidden_state_dim=hidden_state_dim, set2vec_steps=10, set2vec_num_attention_heads=1)

        tf.reset_default_graph()
        input_ph = tf.placeholder(tf.float32, [batch_size, num_atoms, hidden_state_dim])
        mask_ph = tf.placeholder(tf.float32, [batch_size, num_atoms])
        output_vector = Set2Vec(hparams).forward(input_ph, mask_ph)

        input_np = np.random.randn(batch_size, num_atoms, hidden_state_dim)
        input_np_perm = input_np[:, np.random.permutation(num_atoms), :]
        mask_np = np.ones([batch_size, num_atoms])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_vector, feed_dict={input_ph: input_np, mask_ph: mask_np})
            out_perm = sess.run(output_vector, feed_dict={input_ph: input_np_perm, mask_ph: mask_np})

        self.assertAllClose(out, out_perm)

    def test_pad_invariance(self):
        """Test whether the set2vec output is invariant to padding of the input mols."""
        num_atoms, batch_size, hidden_state_dim, padding = 4, 3, 5, 2
        hparams = HParams(node_dim=hidden_state_dim, set2vec_steps=12, set2vec_num_attention_heads=1)

        tf.reset_default_graph()
        input_ph = tf.placeholder(tf.float32, [None, None, hidden_state_dim])
        mask = tf.placeholder(tf.bool, [None, None])
        output_vector = Set2Vec(hparams).forward(input_ph, mask=mask)

        input_np = np.random.randn(batch_size, num_atoms, hidden_state_dim)
        input_np_pad = np.pad(input_np, ((0, 0), (0, padding), (0, 0)), mode='constant')
        mask_np = np.ones((batch_size, num_atoms))
        mask_np_pad = np.pad(mask_np, ((0, 0), (0, padding)), mode='constant')

        # Permute the masks and inputs for each element in the batch.
        # We create separate permutation for each element in order to make the
        # test more general.
        for i in range(batch_size):
            perm = np.random.permutation(mask_np_pad.shape[1])
            mask_np_pad[i, :] = mask_np_pad[i, perm]
            input_np_pad[i, :] = input_np_pad[i, perm]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_original = sess.run(output_vector, feed_dict={input_ph: input_np, mask: mask_np})
            out_pad = sess.run(output_vector, feed_dict={input_ph: input_np_pad, mask: mask_np_pad})
        self.assertAllClose(out_original, out_pad)

    def test_multi_head_attention(self):
        """Test error handling and permutation invariance for multi-head attention"""
        num_atoms, batch_size, hidden_state_dim = 4, 3, 8

        with self.assertRaises(ValueError):
            tf.reset_default_graph()
            input_ph = tf.placeholder(tf.float32, [batch_size, num_atoms, hidden_state_dim])
            mask_ph = tf.placeholder(tf.float32, [batch_size, num_atoms])
            hparams = HParams(node_dim=hidden_state_dim, set2vec_steps=12, set2vec_num_attention_heads=3)
            Set2Vec(hparams).forward(input_ph, mask_ph)

        hparams = HParams(node_dim=hidden_state_dim, set2vec_steps=12, set2vec_num_attention_heads=2)
        tf.reset_default_graph()
        input_ph = tf.placeholder(tf.float32, [batch_size, num_atoms, hidden_state_dim])
        mask_ph = tf.placeholder(tf.float32, [batch_size, num_atoms])
        output_vector = Set2Vec(hparams).forward(input_ph, mask_ph)

        input_np = np.random.randn(batch_size, num_atoms, hidden_state_dim)
        input_np_perm = input_np[:, np.random.permutation(num_atoms), :]
        mask_np = np.ones([batch_size, num_atoms])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_vector, feed_dict={input_ph: input_np, mask_ph: mask_np})
            out_perm = sess.run(output_vector, feed_dict={input_ph: input_np_perm, mask_ph: mask_np})

        self.assertAllClose(out, out_perm)


if __name__ == '__main__':
    tf.test.main()
