import tensorflow as tf
import numpy as np
import numpy.testing as npt
from data.molecules import TFMolBatch, NPMol

# sample molecules
coordinates = np.array([[[0, 0, 0], [1, 0, 0], [-1, 0, 0]],
                        [[0, 0, 0], [1, 1, 1], [1, 0, 0]]])
distance_matrix = np.array([[[0, 1, 1], [1, 0, 2], [1, 2, 0]],
                            [[0, np.sqrt(3), 1], [np.sqrt(3), 0, np.sqrt(2)], [1, np.sqrt(2), 0]]])
distances = np.array([[1, 1, 2], [np.sqrt(3), 1, np.sqrt(2)]])
num_atoms = 3
num_distances = int(num_atoms * (num_atoms - 1) / 2)
batch_size = 2
num_atom_types = 5
atom_types = [['C', 'C', 'C'], ['C', 'C', 'C']]
atoms = np.zeros([batch_size, num_atoms, num_atom_types])
atoms[:, :, -1] = [0.3, 0.2, 0.1]
atoms[:, :, 0] = 1 - atoms[:, :, -1]
mask = np.ones([batch_size, num_atoms])
mask[:, :] = [0.7, 0.8, 0.9]
num_labels = 2
labels = np.random.randn(batch_size, num_labels)


class TestTFMolBatch(tf.test.TestCase):
    def test_distances_provided(self):
        atoms_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atom_types])
        distances_ph = tf.placeholder(tf.float32, shape=[None, num_distances])

        mols = TFMolBatch(atoms=atoms_ph, distances=distances_ph)
        with self.assertRaises(AttributeError):
            _ = mols.coordinates
        distance_matrix_tf = mols.distance_matrix

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            distance_matrix_np = sess.run(distance_matrix_tf, feed_dict={distances_ph: distances, atoms_ph: atoms})
            self.assertAllClose(distance_matrix_np, distance_matrix)

    def test_distance_matrix_provided(self):
        atoms_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atom_types])
        distance_matrix_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atoms])
        mols = TFMolBatch(atoms=atoms_ph, distance_matrix=distance_matrix_ph)
        with self.assertRaises(AttributeError):
            _ = mols.coordinates

        distances_tf = mols.distances
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            distances_np = sess.run(distances_tf, feed_dict={distance_matrix_ph: distance_matrix, atoms_ph: atoms})
            self.assertAllClose(distances_np, distances)

    def test_coordinates_provided(self):
        atoms_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atom_types])
        coordinates_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, 3])
        mols_1 = TFMolBatch(atoms=atoms_ph, coordinates=coordinates_ph)
        mols_2 = TFMolBatch(atoms=atoms_ph, coordinates=coordinates_ph)

        distance_matrix_1_tf = mols_1.distance_matrix
        distances_1_tf = mols_1.distances
        distances_2_tf = mols_2.distances
        distance_matrix_2_tf = mols_2.distance_matrix

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            dist_1_np, dist_2_np, dist_mat_1_np, dist_mat_2_np = sess.run(
                [distances_1_tf, distances_2_tf, distance_matrix_1_tf, distance_matrix_2_tf],
                feed_dict={coordinates_ph: coordinates, atoms_ph: atoms})
            self.assertAllClose(dist_1_np, distances)
            self.assertAllClose(dist_2_np, distances)
            self.assertAllClose(dist_mat_1_np, distance_matrix)
            self.assertAllClose(dist_mat_2_np, distance_matrix)

    def test_none_provided(self):
        atoms_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atom_types])
        with self.assertRaises(ValueError):
            _ = TFMolBatch(atoms=atoms_ph)

        coordinates_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, 3])
        mols = TFMolBatch(atoms=atoms_ph, coordinates=coordinates_ph)
        with self.assertRaises(AttributeError):
            _ = mols.mask

    def test_labels(self):
        atoms_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atom_types])
        coordinates_ph = tf.placeholder(tf.float32, shape=[None, num_atoms, 3])
        labels_ph = tf.placeholder(tf.float32, shape=[None, num_labels])
        mols = TFMolBatch(atoms=atoms_ph, coordinates=coordinates_ph)
        with self.assertRaises(AttributeError):
            _ = mols.labels

        mols = TFMolBatch(atoms=atoms_ph, coordinates=coordinates_ph, labels=labels_ph)
        labels_tf = mols.labels

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_np = sess.run(labels_tf, feed_dict={coordinates_ph: coordinates, atoms_ph: atoms, labels_ph: labels})
            self.assertAllClose(labels_np, labels)


class TestNPMol(npt.TestCase):
    def test_none_provided(self):
        with self.assertRaises(ValueError):
            _ = NPMol(atoms=atoms)

    def test_distances_provided(self):
        mol = NPMol(atoms=atoms[0], distances=distances[0])
        self.assertTrue(isinstance(mol.smiles, str))
        npt.assert_allclose(mol.distance_matrix, distance_matrix[0])
        npt.assert_equal(mol.coordinates.shape, coordinates[0].shape)

    def test_distance_matrix_provided(self):
        mol = NPMol(atoms=atoms[0], distance_matrix=distance_matrix[0])
        npt.assert_allclose(mol.distances, distances[0])
        self.assertTrue(isinstance(mol.smiles, str))
        npt.assert_equal(mol.coordinates.shape, coordinates[0].shape)

    def test_coordinates_provided(self):
        mol = NPMol(atoms=atoms[0], coordinates=coordinates[0])
        npt.assert_allclose(mol.distances, distances[0])
        npt.assert_allclose(mol.distance_matrix, distance_matrix[0])
        self.assertTrue(isinstance(mol.smiles, str))

    def test_atom_types(self):
        mol = NPMol(atoms=atoms[0], coordinates=coordinates[0])
        npt.assert_equal(mol.atoms, atom_types[0])

    def test_padding_removal(self):
        p_coordinates = np.concatenate((coordinates[0], [[0, 0, 0]]))
        p_distance_matrix = np.pad(distance_matrix[0], ((0, 1), (0, 1)), mode='constant')
        p_distances = p_distance_matrix[np.triu_indices(4, 1)]
        p_atoms = np.concatenate((atoms[0], [[0, 0, 0, 0, 1]]))

        mol = NPMol(atoms=p_atoms, coordinates=p_coordinates, distance_matrix=p_distance_matrix,
                    distances=p_distances)
        npt.assert_allclose(mol.distances, distances[0])
        npt.assert_allclose(mol.distance_matrix, distance_matrix[0])
        npt.assert_allclose(mol.coordinates, coordinates[0])
        npt.assert_equal(mol.atoms, atom_types[0])

    def test_batch_construction(self):
        mol_batch = NPMol.create_from_batch(batch_atoms=atoms, batch_distances=distances)
        for i in range(len(distances)):
            npt.assert_allclose(mol_batch[i].distances, distances[i])
            npt.assert_allclose(mol_batch[i].distance_matrix, distance_matrix[i])
            npt.assert_equal(mol_batch[i].atoms, atom_types[i])
            self.assertTrue(isinstance(mol_batch[i].smiles, str))

    def test_empty(self):
        atoms_empty = np.zeros([batch_size, num_atoms, num_atom_types])
        atoms_empty[:, :, -1] = 1
        mol_batch = NPMol.create_from_batch(batch_atoms=atoms_empty, batch_distances=distances)
        for mol in mol_batch:
            npt.assert_equal(mol.distance_matrix, np.zeros([0, 0]))
            npt.assert_equal(mol.coordinates, np.zeros([0, 3]))
            npt.assert_equal(mol.smiles, '')
            npt.assert_equal(mol.atoms, [])
