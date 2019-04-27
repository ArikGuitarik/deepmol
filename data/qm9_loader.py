import numpy as np
import os
import rdkit.Chem as Chem
import tensorflow as tf
from .standardization import Standardization


class QM9Loader:
    """Provide the QM9 data set as a tf.data.Dataset, given an sdf file for molecules and csv file for labels.

    :param data_dir: The directory containing the QM9 files *.sdf, *_labels.csv (*=[training|validation|test])
    :param property_names: List of names of the properties/labels that should be used. If None, all are used.
    :param featurizer: Instance of a subclass of Featurizer that specifies how the mols should be featurized.
    :param standardize_labels: If True, labels will be standardized to zero mean and unit variance.
    """

    all_property_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'energy_U0',
                          'energy_U', 'enthalpy_H', 'free_G', 'Cv', 'energy_U0_atom', 'energy_U_atom',
                          'enthalpy_H_atom', 'free_G_atom']

    def __init__(self, data_dir, featurizer, property_names=None, standardize_labels=True):
        max_num_atoms = featurizer.max_num_atoms
        self.featurizer = featurizer

        self._train_mols, self._train_labels = self._load_data_for_partition(data_dir, 'training', property_names)
        self._val_mols, self._val_labels = self._load_data_for_partition(data_dir, 'validation', property_names)
        self._test_mols, self._test_labels = self._load_data_for_partition(data_dir, 'test', property_names)
        self._standardization = None
        if standardize_labels:
            self._standardization = Standardization()
            self._train_labels = self.standardization.apply(self._train_labels)
            self._val_labels = self.standardization.apply(self._val_labels)
            self._test_labels = self.standardization.apply(self._test_labels)

        self.shapes = {'atoms': [max_num_atoms, featurizer.num_atom_features],
                       'interactions': [max_num_atoms, max_num_atoms, featurizer.num_interaction_features],
                       'coordinates': [max_num_atoms, 3], 'labels': [self._train_labels.shape[1]]}

    @property
    def train_data(self):
        """Create a tf.data.Dataset of the training set."""
        return self._create_tf_dataset(self._train_mols, self._train_labels)

    @property
    def val_data(self):
        """Create a tf.data.Dataset of the validation set."""
        return self._create_tf_dataset(self._val_mols, self._val_labels)

    @property
    def test_data(self):
        """Create a tf.data.Dataset of the test set."""
        return self._create_tf_dataset(self._test_mols, self._test_labels)

    @property
    def standardization(self):
        """Return the Standardization that was used to standardize the data.

        :raises AttributeError: in case the labels have not been standardized
        """
        if self._standardization is None:
            raise AttributeError('Labels have not been standardized.')
        return self._standardization

    def _load_data_for_partition(self, data_dir, partition='training', property_names=None):
        """Load rdkit molecules and labels for the specified partition of the data set.

        :param data_dir: The directory containing the QM9 files *.sdf, *_labels.csv (*=[training|validation|test])
        :param partition: [training|validation|test]
        :param property_names: List of names of the properties/labels that should be used. If None, all are used.
        :return:
            - rdkit molecules as a SDMolSupplier (generator reading from sdf file)
            - ndarray of labels, shaped [num_molecules, labels]
        """
        mol_path = os.path.join(data_dir, partition + '.sdf')
        label_path = os.path.join(data_dir, partition + '_labels.csv')
        mols = Chem.SDMolSupplier(mol_path, removeHs=self.featurizer.implicit_hydrogen)
        labels = self._import_labels(label_path, property_names)

        return mols, labels

    def _import_labels(self, label_path, property_names=None):
        """Load labels from a csv file.

        :param label_path: path to csv file with labels.
        :param property_names: If provided, only the specified properties will be included.
        :return: ndarray of labels, shaped [num_molecules, labels]
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

    def _create_tf_dataset(self, mols, labels):
        """Create a tf.data.Dataset from the given rdkit mols and labels.
        :param mols: rdkit molecules
        :param labels: ndarray of labels
        :return: tf.data.Dataset
        """
        featurizer = self.featurizer

        def generator():
            for mol, label in zip(mols, labels):
                featurized_mol = featurizer.featurize(mol)
                yield {'atoms': featurized_mol.atom_features, 'interactions': featurized_mol.interaction_matrix,
                       'coordinates': featurized_mol.coordinates, 'labels': label}

        return tf.data.Dataset.from_generator(generator,
                                              output_types=({'atoms': tf.float32, 'interactions': tf.float32,
                                                             'coordinates': tf.float32, 'labels': tf.float32}),
                                              output_shapes=self.shapes)
