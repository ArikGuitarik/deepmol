import numpy.testing as npt
import unittest
import rdkit.Chem as Chem
import os
import numpy as np
from data.featurizer import DistanceFeaturizer

max_num_atoms, max_num_atoms_h = 9, 29  # implicit hydrogen, explicit hydrogen


class TestDistanceFeaturizer(unittest.TestCase):
    def setUp(self):
        data_import_directory = os.path.dirname(__file__)
        sdf_path = os.path.join(data_import_directory, '100_sample_molecules.sdf')

        mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)  # import molecules from sdf file
        self.rdkit_mol = mol_supplier[9]
        distance_featurizer = DistanceFeaturizer(max_num_atoms, implicit_hydrogen=True)
        self.feat_mol = distance_featurizer.featurize(self.rdkit_mol)

        mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)  # explicit hydrogen
        rdkit_mol_h = mol_supplier[9]
        distance_featurizer_h = DistanceFeaturizer(max_num_atoms_h, implicit_hydrogen=False)
        self.feat_mol_h = distance_featurizer_h.featurize(rdkit_mol_h)

    def test_max_num_atoms(self):
        distance_featurizer_fail = DistanceFeaturizer(max_num_atoms=1, implicit_hydrogen=True)
        with self.assertRaises(ValueError):
            distance_featurizer_fail.featurize(self.rdkit_mol)

    def test_atom_features(self):
        desired = np.zeros([max_num_atoms, DistanceFeaturizer.num_atom_features])
        desired[0, 0:2] = 3, 1
        desired[1, 1], desired[2, 2] = 1, 1
        npt.assert_equal(self.feat_mol.atom_features, desired)

        desired_h = np.zeros([max_num_atoms_h, DistanceFeaturizer.num_atom_features])
        desired_h[3:6, 0] = 1
        desired_h[0:2, 1] = 1
        desired_h[2, 2] = 1
        npt.assert_equal(self.feat_mol_h.atom_features, desired_h)

    def test_interaction_features(self):
        desired = np.zeros([max_num_atoms, max_num_atoms, DistanceFeaturizer.num_interaction_features])
        desired[0, 1:3, 0] = 1.45685382, 2.61188178
        desired[1, 0:3, 0] = 1.45685382, 0, 1.15502801
        desired[2, 0:2, 0] = 2.61188178, 1.15502801

        npt.assert_allclose(self.feat_mol.interaction_matrix, desired)

        desired_shape_h = [max_num_atoms_h, max_num_atoms_h, DistanceFeaturizer.num_interaction_features]
        npt.assert_equal(self.feat_mol_h.interaction_matrix.shape, desired_shape_h)

    def test_mask(self):
        desired = np.zeros([max_num_atoms])
        desired[0:3] = 1
        npt.assert_equal(self.feat_mol.mask, desired)

        desired_h = np.zeros([max_num_atoms_h])
        desired_h[0:6] = 1
        npt.assert_equal(self.feat_mol_h.mask, desired_h)


if __name__ == '__main__':
    unittest.main()
