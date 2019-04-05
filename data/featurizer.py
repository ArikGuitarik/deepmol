import numpy as np
import rdkit.Chem as Chem


class FeaturizedMolecule:
    """Molecule object holding all the relevant features.

    Args:
        atom_features: atom representations of dimension [num_atoms, num_atom_features]
        interaction_matrix: matrix of atomic interactions, shape: [num_atoms, num_atoms, num_interaction_features]
        mask: vector of length num_atoms indicating whether an atom is actually present (1) or zero-padded (0).
        coordinates: xyz-coordinates of the atoms, shaped [num_atoms, 3]
    """

    def __init__(self, atom_features, interaction_matrix, mask, coordinates=None):
        self.atom_features = atom_features
        self.interaction_matrix = interaction_matrix
        self.mask = mask
        self.coordinates = coordinates


class Featurizer:
    """Base class for featurizers that convert an rdkit molecule into a FeaturizedMolecule.

    Args:
        max_num_atoms: Maximum number of atoms in a molecule. Smaller molecules are zero-padded.
        implicit_hydrogen: True if hydrogen atoms should be treated implicitly.
    """

    def __init__(self, max_num_atoms, implicit_hydrogen=False):
        self.max_num_atoms = max_num_atoms
        self.implicit_hydrogen = implicit_hydrogen

    def featurize(self, rdkit_mol, shuffle_atoms=False):
        """Convert an rdkit_mol to a FeaturizedMolecule.

        Args:
            rdkit_mol: Molecule to be featurized.
            shuffle_atoms: If true, the order of atoms is shuffled.

        Returns:
            FeaturizedMolecule with the respective representations specified in the subclasses.

        Raises:
            ValueError: If the number of atoms exceeds max_num_atoms.
        """
        num_atoms = rdkit_mol.GetNumAtoms()
        if num_atoms > self.max_num_atoms:
            raise ValueError('max_num_atoms is %d, but molecule has %d atoms.' % (self.max_num_atoms, num_atoms))

        if shuffle_atoms:
            order = list(range(num_atoms))
            np.random.shuffle(order)
            rdkit_mol = Chem.RenumberAtoms(rdkit_mol, newOrder=order)

        atom_features = self._generate_atom_features(rdkit_mol)
        interaction_matrix = self._generate_interaction_matrix(rdkit_mol)
        mask = self._generate_mask(rdkit_mol)
        coordinates = self._generate_coordinates(rdkit_mol)

        return FeaturizedMolecule(atom_features, interaction_matrix, mask, coordinates)

    def _generate_atom_features(self, rdkit_mol):
        raise NotImplementedError('Featurizer must be implemented in child class')

    def _generate_interaction_matrix(self, rdkit_mol):
        raise NotImplementedError('Featurizer must be implemented in child class')

    def _generate_mask(self, rdkit_mol):
        raise NotImplementedError('Featurizer must be implemented in child class')

    def _generate_coordinates(self, rdkit_mol):
        """Read the atomic coordinates (shape: [num_atoms, 3]) saved in the rdkit_mol. """
        try:
            conformer = rdkit_mol.GetConformer()  # necessary to access position information
        except ValueError:
            return None  # no coordinates provided in the mol

        num_atoms = rdkit_mol.GetNumAtoms()
        coordinates = np.zeros([self.max_num_atoms, 3])
        for i in range(num_atoms):
            position = conformer.GetAtomPosition(i)
            coordinates[i, :] = position.x, position.y, position.z

        return coordinates
