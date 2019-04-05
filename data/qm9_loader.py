import numpy as np
import rdkit.Chem as Chem
import tensorflow as tf
from data.featurizer import DistanceFeaturizer


class QM9Loader:
    """Provide the QM9 data set as a tf.data.Dataset, given an sdf file for molecules and csv file for labels.

    Args:
        mol_path: Path of the sdf file containing the molecules
        label_path: Path of the csv file containing the labels (in the same order as the molecules)
        property_names: List of names of the properties/labels that should be used. If None, all are used.
        featurizer: Name of the featurizer that determines how a molecule's features should be calculated
        implicit_hydrogen: If True, hydrogen atoms in the molecule will be implicit.
        label_standardization: Provide Standardization to transform labels to zero mean and unit variance.

    Raises:
        ValueError: If the featurizer (provided by name) does not exist.
    """

    all_property_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'energy_U0',
                          'energy_U', 'enthalpy_H', 'free_G', 'Cv', 'energy_U0_atom', 'energy_U_atom',
                          'enthalpy_H_atom', 'free_G_atom']

    def __init__(self, mol_path, label_path, property_names=None, featurizer='distance', implicit_hydrogen=False,
                 shuffle_atoms=False, label_standardization=None):
        self.mol_supplier = Chem.SDMolSupplier(mol_path, removeHs=implicit_hydrogen)  # import molecules from sdf file
        self.shuffle_atoms = shuffle_atoms

        self.labels = self._import_labels(label_path, property_names)
        if label_standardization is not None:
            self.labels = label_standardization.apply(self.labels)

        max_num_atoms = 9 if implicit_hydrogen else 29  # used for padding in QM9 data set
        if featurizer == 'distance':
            self.featurizer = DistanceFeaturizer(max_num_atoms, implicit_hydrogen)
        else:
            raise ValueError('Unknown Featurizer: ' + featurizer)

        self.shapes = {'atoms': [max_num_atoms, self.featurizer.num_atom_features],
                       'interactions': [max_num_atoms, max_num_atoms, self.featurizer.num_interaction_features],
                       'mask': [max_num_atoms], 'coordinates': [max_num_atoms, 3], 'labels': [self.labels.shape[1]]}

    def _import_labels(self, label_path, property_names=None):
        """Load labels from a csv file.

        Args:
            label_path: path to csv file with labels.
            property_names: If provided, only the specified properties will be included.

        Returns:
            ndarray of labels, shaped [num_molecules, labels]
        """
        if property_names is None:
            columns_to_use = np.arange(1, 20)  # all columns except ID
        else:
            label_indices = np.array([self.all_property_names.index(name) for name in property_names])
            columns_to_use = label_indices + 1  # add one since column 0 is id
        labels = np.loadtxt(label_path, delimiter=',', skiprows=1, usecols=columns_to_use)
        # ensure that labels is 2D even if only one property is requested
        labels = labels.reshape([-1, len(columns_to_use)])
        return labels

    def _mol_generator(self):
        """Generator for dataset creation (see create_tf_dataset) """
        for mol, label in zip(self.mol_supplier, self.labels):
            featurized_mol = self.featurizer.featurize(mol, shuffle_atoms=self.shuffle_atoms)
            yield {'atoms': featurized_mol.atom_features, 'interactions': featurized_mol.interaction_matrix,
                    'mask': featurized_mol.mask, 'coordinates': featurized_mol.coordinates, 'labels': label}

    def create_tf_dataset(self):
        """Create a tf.data.Dataset from the imported QM9 data."""
        return tf.data.Dataset.from_generator(self._mol_generator, output_types=(
            {'atoms': tf.float32, 'interactions': tf.float32, 'mask': tf.int32, 'coordinates': tf.float32,
             'labels': tf.float32}), output_shapes=self.shapes)
