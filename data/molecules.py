import numpy as np
import tensorflow as tf
import logging
import pybel
from sklearn.manifold import MDS


class AbstractMol:
    """Abstract class for geometry-encoded molecules that offer internal format conversion.

    The geometry can be represented by coordinates, distance matrix or a flattened vector of the relevant distances.
    If only one representation is provided, but another one is called, conversion is done internally, as defined
    in the concrete subclass.
    If multiple representations are provided at construction, no cross-validation is performed.

    Exact requirements for parameters are specified in the subclasses.
    :param atoms: An encoding of the atom types in the molecule.
    :param distances: the n(n-1)/2 relevant entries of the distance matrix as a flattened tensor
    :param distance_matrix: matrix specifying the interatomic distances
    :param coordinates: cartesian coordinates of the atoms
    :param labels: ndarray of training labels for property prediction
    :raises ValueError: If distances, distance matrix and coordinates are all missing.
    """

    def __init__(self, atoms, distances=None, distance_matrix=None, coordinates=None, labels=None):
        if all([distances is None, distance_matrix is None, coordinates is None]):
            raise ValueError('Either distances, distance_matrix or coordinates need to be specified.')
        self.atoms = atoms
        self._distances = distances
        self._distance_matrix = distance_matrix
        self._coordinates = coordinates
        self._labels = labels

    @property
    def distance_matrix(self):
        """If necessary and possible, it will be generated from distances or coordinates."""
        if self._distance_matrix is None:
            self._generate_distance_matrix()
        return self._distance_matrix

    @property
    def distances(self):
        """If necessary and possible, it will be generated from distance matrix or coordinates."""
        if self._distances is None:
            self._generate_distances()
        return self._distances

    @property
    def coordinates(self):
        """If necessary and possible, they will be generated from the distance matrix."""
        if self._coordinates is None:
            self._generate_coordinates()
        return self._coordinates

    @property
    def labels(self):
        """Return labels if provided at construction.
        :raises AttributeError: If labels have not been provided at construction.
        """
        if self._labels is None:
            raise AttributeError('Labels have not been provided and can not be generated.')
        return self._labels

    def _generate_distance_matrix(self):
        raise NotImplementedError('Must be implemented in child class')

    def _generate_distances(self):
        raise NotImplementedError('Must be implemented in child class')

    def _generate_coordinates(self):
        raise NotImplementedError('Must be implemented in child class')


class TFMolBatch(AbstractMol):
    """Holds TensorFlow tensors representing an entire batch of molecules.

    Different representations of the geometry of the molecule (distance matrix, flattened distances, coordinates)
    are available and internally converted on request. All operations are performed on the whole batch.
    At least one representation is required for atoms and geometry each.

    :param atoms: tensor holding atom features, shaped [batch_size, num_atoms, num_atom_features]
    :param atoms_logits: tensor holding logits of the atom features, shaped [batch_size, num_atoms, num_atom_features].
        atoms are generated from atoms_logits, but not the other way round.
    :param mask: indicates whether an atom is actually present (1) or zero-padded (0), shaped [batch_size, num_atoms]
    :param distances: tensor of relevant entries in the distance matrix [batch_size, n(n-1)/2]
    :param distance_matrix: distance matrices of all molecules in the batch, shaped [batch_size, n, n]
    :param coordinates: coordinates of all atoms in each molecule in the batch, shaped [batch_size, n, 3]
    :param labels: ndarray of training labels, shaped [batch_size, num_properties]
    :raises ValueError: If no representation for atoms or geometry is provided.
    """

    def __init__(self, atoms=None, atoms_logits=None, mask=None, distances=None, distance_matrix=None, coordinates=None,
                 labels=None):
        if atoms is None:
            if atoms_logits is None:
                raise ValueError('Either atoms or atoms_logits need to be provided.')
            else:
                atoms = tf.nn.softmax(atoms_logits)
        self.atoms_logits = atoms_logits
        self._mask = mask

        super(TFMolBatch, self).__init__(atoms, distances, distance_matrix, coordinates, labels)

    @property
    def mask(self):
        """ Return the mask. If necessary, it will be generated from the values for atom type NONE. """
        if self._mask is None:
            self._generate_mask()
        return self._mask

    def _generate_mask(self):
        self._mask = 1 - self.atoms[:, :, -1]

    def _generate_coordinates(self):
        """Coordinates can not be generated from distance matrix or distances."""
        raise AttributeError('Coordinates have not been provided and can not be generated.')

    def _generate_distances(self):
        """Generate the flattened distances from either distance matrix or coordinates.
        :raises AttributeError: if neither distance matrix nor coordinates are given.
        """
        if self._distance_matrix is not None:
            self._distance_matrix_to_distances()
        elif self._coordinates is not None:
            self._coordinates_to_distances()
        else:
            raise AttributeError('Could not generate distances: Neither distance matrix nor coordinates are given.')

    def _distance_matrix_to_distances(self):
        """Generate flattened distance vector from distance matrix using tf.gather_nd"""
        # To use tf.gather_nd, we need to specify indices we want to extract from the input matrix.
        # Numpy provides us with the indices of the upper triangle of a 2D matrix. -> [row, column]
        num_atoms = self.atoms.get_shape()[1].value
        upper_triangle_indices = np.dstack(np.triu_indices(num_atoms, 1)).reshape([-1, 2])
        num_matrix_elements = upper_triangle_indices.shape[0]

        with tf.name_scope('dist_matrix_to_dist'):
            # Due to the additional batch dimension, the input matrix is a 3D matrix. Thus, the specified indices need
            # to be of the form [batch_element, row, column].
            # At first, we provide one set of upper triangle indices for each element in the batch.
            batch_size = tf.shape(self.distance_matrix)[0]
            tiled_indices = tf.tile(upper_triangle_indices, [batch_size, 1])
            tiled_indices = tf.reshape(tiled_indices, [batch_size, num_matrix_elements, 2])
            tiled_indices = tf.to_int32(tiled_indices)

            # Then, we add the batch index everywhere
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.tile(tf.reshape(batch_indices, [batch_size, 1]), [1, num_matrix_elements])
            batch_indices = tf.expand_dims(batch_indices, 2)

            all_indices = tf.concat([batch_indices, tiled_indices], axis=2)

            self._distances = tf.gather_nd(params=self.distance_matrix, indices=all_indices)

    def _generate_distance_matrix(self):
        """Generate distance matrix from flattened vector of interatomic distances using tf.gather_nd."""
        distances = self.distances  # property call ensures that distances has been generated
        num_atoms = self.atoms.get_shape()[1].value
        upper_triangle_indices = np.dstack(np.triu_indices(num_atoms, 1)).reshape([-1, 2])

        inverse_indices = np.zeros([num_atoms, num_atoms])
        for i, indices in enumerate(upper_triangle_indices):
            inverse_indices[indices[0], indices[1]] = i
            inverse_indices[indices[1], indices[0]] = i

        with tf.name_scope('dist_to_dist_matrix'):
            batch_size = tf.shape(distances)[0]
            tiled_indices = tf.tile(inverse_indices, [batch_size, 1])
            tiled_indices = tf.reshape(tiled_indices, [batch_size, num_atoms, num_atoms, 1])
            tiled_indices = tf.to_int32(tiled_indices)

            batch_indices = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.reshape(batch_indices, [batch_size, 1])
            batch_indices = tf.tile(batch_indices, [1, num_atoms ** 2])
            batch_indices = tf.reshape(batch_indices, [batch_size, num_atoms, num_atoms, 1])

            all_indices = tf.concat([batch_indices, tiled_indices], axis=-1)

            distance_matrix = tf.gather_nd(params=distances, indices=all_indices)

            # set diagonal to zero
            mask = tf.ones([num_atoms, num_atoms]) - tf.eye(num_atoms)
            self._distance_matrix = distance_matrix * mask

    def _coordinates_to_distances(self):
        """Generate flattened distance vector from coordinates."""
        # first gather all pairs of points using tf.gather_nd()
        with tf.name_scope('coord_to_dist'):
            batch_size = tf.shape(self.coordinates)[0]
            num_atoms = self.atoms.get_shape()[1].value
            num_distances = int(num_atoms * (num_atoms - 1) / 2)

            indices = np.zeros([num_distances, 2, 3, 2])  # dist, 2 points to compare, xyz, indices in coordinates
            counter = 0
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    indices[counter, 0, 0, :] = [i, 0]
                    indices[counter, 0, 1, :] = [i, 1]
                    indices[counter, 0, 2, :] = [i, 2]
                    indices[counter, 1, 0, :] = [j, 0]
                    indices[counter, 1, 1, :] = [j, 1]
                    indices[counter, 1, 2, :] = [j, 2]
                    counter += 1

            batch_indices = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.tile(tf.reshape(batch_indices, [batch_size, 1, 1, 1, 1]),
                                    [1, num_distances, 2, 3, 1])

            coordinate_indices = tf.tile(indices, [batch_size, 1, 1, 1])
            coordinate_indices = tf.reshape(coordinate_indices, [batch_size, num_distances, 2, 3, 2])
            coordinate_indices = tf.to_int32(coordinate_indices)
            coordinate_indices = tf.concat([batch_indices, coordinate_indices], axis=-1)

            point_pairs = tf.gather_nd(params=self.coordinates, indices=coordinate_indices)

            # calculate Euclidian distances
            distances = (point_pairs[:, :, 0, :] - point_pairs[:, :, 1, :]) ** 2
            distances = tf.reduce_sum(distances, axis=-1)
            distances = tf.sqrt(distances)
            self._distances = distances


class NPMol(AbstractMol):
    """Holds np.ndarrays representing a single molecule, provided in different useful formats.

    At construction, converts from a zero-padded, stochastic reconstruction to a molecule with definite values.
    Moreover, provides internal conversion between different formats: distance matrix, flattened distances, coordinates,
    smiles, xyz file format

    For construction, atoms and [distances|distance_matrix|coordinates] are necessary.

    :param atoms: one-hot vector or probability distribution over atom types, will be converted to a
        list of atom symbols at construction.
    :param distances: flattened ndarray of n(n-1)/2 relevant entries in the distance matrix
    :param distance_matrix: The distance matrix of the molecule.
    :param coordinates: The 3D coordinates for all atoms in the molecule.
    :param smiles: Optionally, provide the SMILES string of the molecule.
    """

    def __init__(self, atoms, distances=None, distance_matrix=None, coordinates=None, smiles=None):
        super(NPMol, self).__init__(atoms, distances=distances, distance_matrix=distance_matrix,
                                    coordinates=coordinates)
        self._smiles = smiles
        self._xyz = None
        self._valid_distances = None
        self._discretize()

    def _discretize(self):
        """Remove padding and convert the probabilistic description of atom types to a concrete list of atom symbols."""
        atom_types, mask = self._get_atom_types_and_mask()
        self.atoms = atom_types  # overwrite atom one-hot-matrix with list of type symbols
        self._remove_zero_padding(mask)

    def _remove_zero_padding(self, mask):
        """Remove zero-padding on the basis of the provided boolean mask which specifies which atoms are present."""
        padded_atom_count = len(mask)
        actual_atom_count = sum(mask)
        mask_matrix = mask * mask.reshape([padded_atom_count, 1])
        # remove coordinates of non-existent atoms
        if self._coordinates is not None:
            self._coordinates = self._coordinates[mask]
        if self._distance_matrix is not None:
            unpadded_distance_matrix = np.reshape(self._distance_matrix[mask_matrix],
                                                  [actual_atom_count, actual_atom_count])
            self._distance_matrix = unpadded_distance_matrix
        if self._distances is not None:
            flattened_mask_matrix = mask_matrix[np.triu_indices(padded_atom_count, 1)]
            self._distances = self._distances[flattened_mask_matrix]

    def _get_atom_types_and_mask(self):
        """Generate list of unambiguous atom symbols and a boolean mask from a (possibly) stochastic atom type matrix.
        The most probable atom type is chosen and converted to its string symbol.
        If atom type NONE is most probable, the atom is not included in the list of symbols,
        and the respective mask value is set to False.
        """
        most_probable_indices = np.argmax(self.atoms, axis=-1)
        mask = most_probable_indices < 4  # 4 = none_index
        atom_count = mask.sum()
        if atom_count == 0:
            logging.warning('Molecule does not contain any atoms.')

        # atom types to strings/symbols
        available_types = ['C', 'N', 'O', 'F']
        filtered_indices: np.core.multiarray.ndarray = most_probable_indices[mask]
        atom_types = [available_types[type_index] for type_index in filtered_indices]

        return atom_types, mask

    @property
    def xyz(self):
        """Atom types and coordinates in the xyz data format, generated from distances or coordinates."""
        if self._xyz is None:
            self._generate_xyz()
        return self._xyz

    @property
    def smiles(self):
        """Canonical SMILES string, generated if necessary."""
        if self._smiles is None:
            self._generate_smiles()
        return self._smiles

    def _generate_xyz(self):
        """XYZ format, generated if necessary."""
        # write in xyz format
        xyz = '{}\n\n'.format(len(self.atoms))
        for i, point in enumerate(self.coordinates):
            xyz += self.atoms[i] + '\t{}\t{}\t{}\n'.format(point[0], point[1], point[2])

        self._xyz = xyz

    def _generate_smiles(self):
        """Generate canonical SMILES strings from atoms and coordinates."""
        if len(self.atoms) == 0:
            self._smiles = ''
            return

        parsed_mol = pybel.readstring('xyz', self.xyz)
        self._smiles = parsed_mol.write('can', opt={'n': None, 'i': None})  # remove mol name and stereochemistry

    def _generate_coordinates(self):
        """Generate atomic coordinates from distance matrix using multi-dimensional scaling."""
        atom_count = self.distance_matrix.shape[0]
        if atom_count > 1:
            mds = MDS(n_components=3, metric=True, dissimilarity='precomputed', eps=1e-7, n_init=1)
            self._coordinates = mds.fit_transform(self.distance_matrix)
        else:
            self._coordinates = np.zeros([atom_count, 3])

    def _generate_distance_matrix(self):
        """Generate distance matrix from coordinates or distances."""
        atom_count = len(self.atoms)
        distance_matrix = np.zeros([atom_count, atom_count])
        if self._distances is not None:
            distance_matrix[np.triu_indices(atom_count, 1)] = self.distances
            distance_matrix += np.transpose(distance_matrix)
            self._distance_matrix = distance_matrix

        elif self._coordinates is not None:
            for i in range(atom_count):
                for j in range(atom_count):
                    difference = self._coordinates[i] - self._coordinates[j]
                    distance = np.sqrt(np.sum(difference * difference))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

            self._distance_matrix = distance_matrix
        else:
            raise AttributeError('Could not generate distance matrix, neither distances nor coordinates are given.')

    def _generate_distances(self):
        """Generate distances from distance matrix."""
        atom_count = self.distance_matrix.shape[0]
        self._distances = self.distance_matrix[np.triu_indices(atom_count, 1)]

    @classmethod
    def create_from_batch(cls, batch_atoms, batch_distances=None, batch_distance_matrix=None, batch_coordinates=None):
        """Create a list of DiscreteMols from data where the first dimension is the batch dimension."""
        if batch_atoms is None:
            raise ValueError('Atoms must be provided.')
        if all([batch_distances is None, batch_distance_matrix is None, batch_coordinates is None]):
            raise ValueError('Either distances, distance_matrix or coordinates need to be specified.')

        batch_size = batch_atoms.shape[0]

        # replace params with value None by a list of Nones, so we can iterate over them and pass on None for each mol.
        none_list = [None for _ in range(batch_size)]
        batch_coordinates = none_list if batch_coordinates is None else batch_coordinates
        batch_distances = none_list if batch_distances is None else batch_distances
        batch_distance_matrix = none_list if batch_distance_matrix is None else batch_distance_matrix

        batch_mols = [cls(atoms, distances=distances, distance_matrix=distance_matrix, coordinates=coordinates)
                      for atoms, coordinates, distances, distance_matrix
                      in zip(batch_atoms, batch_coordinates, batch_distances, batch_distance_matrix)]

        return batch_mols
