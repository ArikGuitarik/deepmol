import numpy as np
import rdkit.Chem as Chem


class FeaturizedMolecule:
    """Molecule object holding all the relevant features.

    :param atom_features: atom representations of dimension [num_atoms, num_atom_features]
    :param interaction_matrix: matrix of atomic interactions shaped [num_atoms, num_atoms, num_interaction_features]
    :param mask: vector of length num_atoms indicating whether an atom is actually present (1) or zero-padded (0).
    :param coordinates: xyz-coordinates of the atoms, shaped [num_atoms, 3]
    """

    def __init__(self, atom_features, interaction_matrix, mask, coordinates=None):
        self.atom_features = atom_features
        self.interaction_matrix = interaction_matrix
        self.mask = mask
        self.coordinates = coordinates


class Featurizer:
    """Base class for featurizers that convert an rdkit molecule into a FeaturizedMolecule.

    :param max_num_atoms: Maximum number of atoms in a molecule. Smaller molecules are zero-padded.
    :param implicit_hydrogen: True if hydrogen atoms should be treated implicitly.
    """

    def __init__(self, max_num_atoms, implicit_hydrogen=False):
        self.max_num_atoms = max_num_atoms
        self.implicit_hydrogen = implicit_hydrogen

    def featurize(self, rdkit_mol, shuffle_atoms=False):
        """Convert an rdkit_mol to a FeaturizedMolecule.

        :param rdkit_mol: Molecule to be featurized.
        :param shuffle_atoms: If True, the order of atoms is shuffled.
        :return: FeaturizedMolecule with the respective representations specified in the subclasses.
        :raises ValueError: If the number of atoms exceeds max_num_atoms.
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


class DistanceFeaturizer(Featurizer):
    """Featurizer based on interatomic distance."""
    num_atom_features = 5
    num_interaction_features = 1

    def _generate_interaction_matrix(self, rdkit_mol):
        """Generate interaction matrix using the real-valued Euclidian distance as interaction feature."""
        num_atoms = rdkit_mol.GetNumAtoms()
        interaction_matrix = np.zeros((num_atoms, num_atoms, self.num_interaction_features))
        interaction_matrix[:, :, 0] = Chem.Get3DDistanceMatrix(rdkit_mol)

        # add zero padding
        padding_size = self.max_num_atoms - num_atoms
        interaction_matrix = np.pad(interaction_matrix, ((0, padding_size), (0, padding_size), (0, 0)), mode='constant')

        return interaction_matrix

    def _generate_atom_features(self, rdkit_mol):
        """Generate atom features by one-hot encoding the atom types.

        If hydrogen is implicit, the entry for hydrogen is replaced by the number of implicit hydrogens.
        """
        atoms = rdkit_mol.GetAtoms()
        atomic_numbers = np.array([1, 6, 7, 8, 9])  # QM9 data set only has HCNOF

        atom_features = np.zeros((len(atoms), self.num_atom_features))
        for i, atom in enumerate(atoms):
            atomic_num = atom.GetAtomicNum()
            atom_type_one_hot = (atomic_numbers == atomic_num).astype(int)  # HCNOF one-hot
            atom_features[i, 0:5] = atom_type_one_hot
            # if hydrogen is implicit, the 1st entry is always 0 -> replace it with number of implicit hydrogen atoms
            if self.implicit_hydrogen:
                atom_features[i, 0] = atom.GetNumImplicitHs()
        # add zero padding
        padding_size = self.max_num_atoms - len(atoms)
        atom_features = np.pad(atom_features, ((0, padding_size), (0, 0)), mode='constant')

        return atom_features

    def _generate_mask(self, rdkit_mol):
        num_atoms = rdkit_mol.GetNumAtoms()
        mask = np.zeros(self.max_num_atoms)
        mask[:num_atoms] = 1

        return mask
