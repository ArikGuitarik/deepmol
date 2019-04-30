import logging
import os
import tensorflow as tf
import numpy as np
import argparse
from train_util import QM9Trainer, ConfigReader
from model.molvae import MolVAE
from data.featurizer import DistanceFeaturizer
from data.molecules import NPMol
import pybel
import openbabel

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


class VAETrainer(QM9Trainer):
    """Extend QM9Trainer to train and evaluate a geometry-based variational autoencoder for molecules.

    :param data_dir: directory containing the QM9 files *.sdf, *_labels.csv (*=[training|validation|test])
    :param train_log_interval: Write training log after this many steps.
    :param val_log_interval: Write validation log after this many steps.
    :param name: Name of the experiment/training that is performed.
    :param implicit_hydrogen: If True, hydrogen atoms will be treated implicitly.
    :param patience: Stop training early if the (smoothed) validation loss has not improved for this many steps.
    :param loss_smoothing: Early stopping is decided based on a running average of the validation loss.
        This parameter controls the amount of smoothing and corresponds to the TensorBoard smoothing slider.
    """
    def __init__(self, data_dir, train_log_interval, val_log_interval, name='', implicit_hydrogen=True,
                 patience=float('inf'), loss_smoothing=0.8):
        super().__init__(data_dir, train_log_interval, val_log_interval, name, implicit_hydrogen,
                         patience, loss_smoothing, property_names=None)

        max_num_atoms = 9 if implicit_hydrogen else 29  # relevant for zero padding
        self.featurizer = DistanceFeaturizer(max_num_atoms, implicit_hydrogen)

        # initialized by _build_model
        self._train_summary = None
        self._val_loss = None
        self._test_loss = None

    def _write_train_log(self):
        """Perform training step and and write training log. Overrides superclass method."""
        summary, _ = self._sess.run([self._train_summary, self._train_op])
        self._summary_writer.add_summary(summary, self._step)

    def _write_val_log(self):
        """Perform validation, write validation log and return validation loss. Overrides superclass method.

        :return: validation loss
        """
        val_loss = self._average_over_dataset(self._val_iterator, self._val_loss)

        summary = tf.Summary()
        summary.value.add(tag='avg_val_loss', simple_value=val_loss)
        self._summary_writer.add_summary(summary, self._step)
        self._summary_writer.flush()

        return val_loss

    def _eval_results(self):
        """Compute scores on validation and test set and write to results file."""
        results = {}
        results['val_loss'] = self._average_over_dataset(self._val_iterator, self._val_loss)
        results['test_loss'] = self._average_over_dataset(self._test_iterator, self._test_loss)

        logging.info('Computing validation accuracies...')
        val_accuracies = self._compute_accuracies(self._val_iterator, self._val_mols, self._val_mols_rec)
        results['val_atom_acc'], results['val_smiles_acc'], results['val_pos_rmse'], results[
            'val_pos_mae'] = val_accuracies

        logging.info('Computing test accuracies...')
        test_accuracies = self._compute_accuracies(self._test_iterator, self._test_mols, self._test_mols_rec)
        results['test_atom_acc'], results['test_smiles_acc'], results['test_pos_rmse'], results[
            'test_pos_mae'] = test_accuracies

        if self._variational:
            logging.info('Sampling molecules...')
            self.sample_mols(num_batches=1)

        self._write_eval_results_to_file(results)

    def sample_mols(self, num_batches):
        """Sample molecules from the latent prior and write them to disk.

        :param num_batches: Number of batches to sample.
        """
        xyz_list = []
        for _ in range(num_batches):
            if self._coordinate_output:
                sampled_atoms, sampled_coords = self._sess.run(
                    [self._sampled_mols.atoms, self._sampled_mols.coordinates])
                sampled_mols = NPMol.create_from_batch(batch_atoms=sampled_atoms, batch_coordinates=sampled_coords)
            else:
                sampled_atoms, sampled_dist = self._sess.run([self._sampled_mols.atoms, self._sampled_mols.distances])
                sampled_mols = NPMol.create_from_batch(batch_atoms=sampled_atoms, batch_distances=sampled_dist)
            batch_xyz = [mol.xyz for mol in sampled_mols]
            xyz_list.extend(batch_xyz)

        xyz_dir = os.path.join(self.results_dir, 'sampled')
        os.makedirs(xyz_dir, exist_ok=True)
        max_num_digits = len(str(len(xyz_list)))
        for i, xyz in enumerate(xyz_list):
            path = os.path.join(xyz_dir, str(i).zfill(max_num_digits) + '.xyz')
            with open(path, 'w') as f:
                f.write(xyz)

    def _compute_accuracies(self, data_iterator, mols, mols_rec):
        """Iterate over the given set to evaluate scores.

        :param data_iterator: The initializable_iterator of the relevant data set.
        :param mols: TFMolBatch of the original molecules from the data set.
        :param mols_rec: TFMolBatch of the reconstructed molecules.
        :return:
            - smiles_accuracy: The ratio of molecules where the SMILES string was correctly reconstructed.
            - atom_accuracy: The ratio of molecules where all atom types were correctly reconstructed.
            - rmse: The root-mean-square error of atom positions for molecules with correctly reconstructed atoms.
            - mae: The mean absolute error of atom positions for molecules with correctly reconstructed atoms.
        """
        self._sess.run(data_iterator.initializer)
        correct_smiles_counter, correct_atoms_counter, total_counter = 0, 0, 0
        mse_sum, mae_sum = 0, 0
        while True:
            try:
                if self._coordinate_output:
                    atoms, coords, atoms_rec, coords_rec = self._sess.run(
                        [mols.atoms, mols.coordinates, mols_rec.atoms, mols_rec.coordinates])
                    mols_np = NPMol.create_from_batch(atoms, batch_coordinates=coords)
                    rec_mols_np = NPMol.create_from_batch(atoms_rec, batch_coordinates=coords_rec)
                else:
                    atoms, dist, atoms_rec, dist_rec = self._sess.run(
                        [mols.atoms, mols.distances, mols_rec.atoms, mols_rec.distances])
                    mols_np = NPMol.create_from_batch(atoms, batch_distances=dist)
                    rec_mols_np = NPMol.create_from_batch(atoms_rec, batch_distances=dist_rec)

                for mol, rec_mol in zip(mols_np, rec_mols_np):
                    if mol.smiles == rec_mol.smiles:
                        correct_smiles_counter += 1
                    if mol.atoms == rec_mol.atoms:
                        correct_atoms_counter += 1
                        mse, mae = self._compute_position_accuracy(mol, rec_mol)
                        mse_sum += mse
                        mae_sum += mae
                    total_counter += 1
            except tf.errors.OutOfRangeError:
                break
        smiles_accuracy = 0
        atom_accuracy = 0
        rmse, mae = -1, -1
        if total_counter != 0:
            smiles_accuracy = correct_smiles_counter / total_counter
        if correct_atoms_counter != 0:
            atom_accuracy = correct_atoms_counter / total_counter
            rmse = np.sqrt(mse_sum / correct_atoms_counter)
            mae /= correct_atoms_counter
        return atom_accuracy, smiles_accuracy, rmse, mae

    @staticmethod
    def _compute_position_accuracy(mol, rec_mol):
        """Compute the scores comparing the atom positions in two molecules.

        The mean squared error and mean absolute error of their atom positions is calculated.
        To do this, rec_mol is aligned with mol using openbabel.
        A mirrored version of rec_mol is used if this leads to a lower MSE.

        :param mol: NPMol
        :param rec_mol: NPMol
        :return:
            - mean squared error of atom positions
            - mean absolute error of atom positions
        """
        py_mol = pybel.readstring('xyz', mol.xyz)
        py_rec_mol = pybel.readstring('xyz', rec_mol.xyz)
        rec_mol.invert_coordinates()  # change chirality
        py_rec_mol_inverted = pybel.readstring('xyz', rec_mol.xyz)

        def align_mols(ref, to_be_aligned):
            """Align a pybel.Molecule (in place) to a reference one.

            :param ref: the reference pybel.Molecule
            :param to_be_aligned: the pybel.Molecule that will be aligned to ref
            """
            ref, to_be_aligned = ref.OBMol, to_be_aligned.OBMol
            align = openbabel.OBAlign(False, False)
            align.SetRefMol(ref)
            align.SetTargetMol(to_be_aligned)
            align.Align()
            align.UpdateCoords(to_be_aligned)

        align_mols(py_mol, py_rec_mol)
        align_mols(py_mol, py_rec_mol_inverted)

        def mse_mae(mol_1, mol_2):
            """Calculate mean squared distance and mean absolute distance between the atoms in mol_1 and mol_2.

            Each atom in mol_1 is compared with its counterpart in mol_2. Thus, the number of atoms must be equal.

            :param mol_1: pybel.Molecule
            :param mol_2: pybel.Molecule
            :return:
                - mean squared error of atom positions
                - mean absolute error of atom positions
            """
            mse_sum, mae_sum, num_atoms = 0, 0, 0
            for atom_1, atom_2 in zip(mol_1.atoms, mol_2.atoms):
                x1, y1, z1 = atom_1.coords
                x2, y2, z2 = atom_2.coords
                squared_dist = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
                mse_sum += squared_dist
                mae_sum += np.sqrt(squared_dist)
                num_atoms += 1

            return mse_sum / num_atoms, mae_sum / num_atoms

        mse, mae = mse_mae(py_mol, py_rec_mol)
        mse_inverted, mae_inverted = mse_mae(py_mol, py_rec_mol_inverted)

        if mse < mse_inverted:
            return mse, mae
        else:
            return mse_inverted, mae_inverted

    def _build_model(self, hparams):
        """Build the VAE model given the hyperparameter configuration. Overrides superclass method.

        :param hparams: tf.contrib.training.HParams object
        """
        vae = MolVAE(hparams, self.featurizer.max_num_atoms, self.featurizer.num_atom_features)
        train_loss = vae.calculate_loss(self._train_mols)
        # TensorBoard Summaries
        tf.summary.scalar('training_loss', train_loss)
        self._train_summary = tf.summary.merge_all()

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(hparams.learning_rate, self._global_step, hparams.lr_decay_steps,
                                                   hparams.lr_decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train_op = optimizer.minimize(train_loss, global_step=self._global_step)

        self._val_loss = vae.calculate_loss(self._val_mols)
        self._test_loss = vae.calculate_loss(self._test_mols)

        self._val_mols_rec = vae.reconstruct(self._val_mols)
        self._test_mols_rec = vae.reconstruct(self._test_mols)

        self._coordinate_output = hparams.coordinate_output
        self._variational = hparams.variational
        self._sampled_mols = vae.sample(hparams.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory containing data and labels for training, validation and test')
    parser.add_argument('--config_dir', help='directory containing json files with HParams')
    parser.add_argument('--steps', type=int, default=3000000, help='number of steps/batches to train', )
    parser.add_argument('--name', default='', help='prefix of results directory')
    parser.add_argument('--train_log_interval', type=int, default=250, help='write train log after this many steps')
    parser.add_argument('--val_log_interval', type=int, default=10000,
                        help='write validation log after this many steps')
    parser.add_argument('--patience', type=float, default=float('inf'),
                        help='early stopping: stop if validation loss has not improved for this number of steps')
    parser.add_argument('--smoothing_factor', type=float, default=0.8, help='smoothing factor for early stopping')
    parser.add_argument('--explicit_hydrogen', help='treat hydrogen atoms explicitly', action='store_true')

    args = parser.parse_args()

    trainer = VAETrainer(args.data_dir, args.train_log_interval, args.val_log_interval, args.name,
                         not args.explicit_hydrogen, args.patience, args.smoothing_factor)
    if args.config_dir is not None:
        config_reader = ConfigReader(args.config_dir, MolVAE.default_hparams())
        new_hparam_configs = config_reader.get_new_hparam_configs()
        while len(new_hparam_configs) > 0:
            logging.info('Found %d new hyperparameter configurations.', len(new_hparam_configs))
            trainer.run_trainings(new_hparam_configs, args.steps)
            new_hparam_configs = config_reader.get_new_hparam_configs()
    else:
        logging.info('No hyperparameter configurations specified. Using default values.')
        trainer.run_trainings({'default': MolVAE.default_hparams()}, args.steps)
