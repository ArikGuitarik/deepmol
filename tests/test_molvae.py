import tensorflow as tf
from model.molvae import MolVAE
import numpy as np
from data.molecules import TFMolBatch

hparams = MolVAE.default_hparams()
hparams.set_hparam('batch_size', 1)
hparams.set_hparam('geometric_penalty_weight', 0.0)


class TestGeometricPenalty(tf.test.TestCase):
    def test_geometric_penalty(self):
        """Create a three-atomic molecule where the interatomic distances violate the triangle inequality."""
        batch_atom_decoded = np.zeros([1, 9, 5])
        batch_atom_decoded[0, 0:3, 0] = 1  # three C atoms
        batch_atom_decoded[0, 3:9, -1] = 1  # rest is set to none/padded

        batch_dist_decoded = np.zeros([1, 36])
        batch_dist_decoded[0, 0:2] = 1  # set two distances values to 1
        batch_dist_decoded[0, 8] = 5  # and the third to 5 -> violation 5 - (1 + 1) = 3

        decoded_atoms_ph = tf.placeholder(tf.float32, [1, 9, 5])
        decoded_dist_ph = tf.placeholder(tf.float32, [1, 36])
        decoded_mols = TFMolBatch(atoms=decoded_atoms_ph, distances=decoded_dist_ph)

        geom_pen = MolVAE(hparams, 9, 5).geometric_penalty(decoded_mols)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_val = sess.run(geom_pen, feed_dict={decoded_dist_ph: batch_dist_decoded,
                                                     decoded_atoms_ph: batch_atom_decoded})
            self.assertAllClose(loss_val, 3.0)


class TestReconstructionLoss(tf.test.TestCase):
    def test_reconstruction_loss(self):
        num_atoms, num_types = 5, 5
        atoms = np.eye(num_types)[np.newaxis, :]
        coords = np.zeros([1, num_atoms, 3])
        coords[0, :, 0] = np.arange(num_atoms)

        p1 = 0.6
        p2 = (1 - p1) / (num_atoms - 1)
        l1, l2 = np.log(p1), np.log(p2)
        rec_atom_logits = (np.ones_like(atoms) - np.eye(num_types)) * l2 + np.eye(num_types) * l1
        rec_coords = np.zeros_like(coords)
        rec_coords[0, :, 0] = np.linspace(0.1, 4.5, num=num_atoms)

        coord_ph = tf.placeholder_with_default(coords.astype(np.float32), coords.shape)
        atom_ph = tf.placeholder_with_default(atoms.astype(np.float32), atoms.shape)
        mols = TFMolBatch(atoms=atom_ph, coordinates=coord_ph)

        rec_coord_ph = tf.placeholder_with_default(rec_coords.astype(np.float32), rec_coords.shape)
        rec_atom_logits_ph = tf.placeholder_with_default(rec_atom_logits.astype(np.float32), rec_atom_logits.shape)
        rec_mols = TFMolBatch(atoms_logits=rec_atom_logits_ph, coordinates=rec_coord_ph)

        atom_loss = - l1 * num_atoms
        dist_loss = 0.2 / 6.0
        gamma = 0.5
        hparams.set_hparam('gamma', gamma)
        reconstruction_loss = MolVAE(hparams, num_atoms, num_types).reconstruction_loss(mols, rec_mols)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_val = sess.run(reconstruction_loss)
            np.testing.assert_allclose(loss_val, gamma * atom_loss + (2 - gamma) * dist_loss, rtol=1e-4)
