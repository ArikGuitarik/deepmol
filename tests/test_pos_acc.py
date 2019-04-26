import unittest
import os
import numpy as np
from train_vae import VAETrainer
import rdkit.Chem as Chem
from data.featurizer import DistanceFeaturizer
from data.molecules import NPMol


class TestPositionAccuracy(unittest.TestCase):
    def test_atom_features(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
        trainer = VAETrainer(data_dir, 10, 10)

        sdf_path = os.path.join(os.path.dirname(__file__), 'sample_mol.sdf')
        mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)  # import molecules from sdf file
        rdkit_mol = mol_supplier[0]

        featurizer = DistanceFeaturizer(9, implicit_hydrogen=True)
        feat_mol = featurizer.featurize(rdkit_mol)

        coordinates = np.copy(feat_mol.coordinates)
        coordinates *= -1  # invert
        coordinates[:, 0] *= -1  # mirror

        mol = NPMol(atoms=feat_mol.atom_features, coordinates=feat_mol.coordinates)
        rec_mol = NPMol(atoms=feat_mol.atom_features, coordinates=coordinates)

        mse, mae = trainer._compute_position_accuracy(mol, rec_mol)
        np.testing.assert_allclose([mse, mae], [0, 0], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
