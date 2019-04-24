import logging
import tensorflow as tf
import argparse
from train_util import QM9Trainer, ConfigReader
from model.molvae import MolVAE
from data.featurizer import DistanceFeaturizer
from data.molecules import NPMol

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


class VAETrainer(QM9Trainer):

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
        results['val_atom_acc'], results['val_smiles_acc'] = val_accuracies

        logging.info('Computing test accuracies...')
        test_accuracies = self._compute_accuracies(self._test_iterator, self._test_mols, self._test_mols_rec)
        results['test_atom_acc'], results['test_smiles_acc'] = test_accuracies

        self._write_eval_results_to_file(results)

    def _compute_accuracies(self, data_iterator, mols, mols_rec):
        """Iterate over the given set to evaluate scores.

        :param data_iterator: The initializable_iterator of the relevant data set.
        :param mols: TFMolBatch of the original molecules from the data set.
        :param mols_rec: TFMolBatch of the reconstructed molecules.
        :return:
            - smiles_accuracy: The ratio of molecules where the SMILES string was correctly reconstructed.
            - atom_accuracy: The ratio of molecules where all atom types were correctly reconstructed.
        """
        self._sess.run(data_iterator.initializer)
        correct_smiles_counter, correct_atoms_counter, total_counter = 0, 0, 0
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
                    total_counter += 1
            except tf.errors.OutOfRangeError:
                break
        smiles_accuracy = 0
        atom_accuracy = 0
        if total_counter != 0:
            smiles_accuracy = correct_smiles_counter / total_counter
        if correct_atoms_counter != 0:
            atom_accuracy = correct_atoms_counter / total_counter
        return atom_accuracy, smiles_accuracy

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory containing data and labels for training, validation and test')
    parser.add_argument('config_dir', help='directory containing json files with HParams')
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

    config_reader = ConfigReader(args.config_dir, MolVAE.default_hparams())
    new_hparam_configs = config_reader.get_new_hparam_configs()
    while len(new_hparam_configs) > 0:
        logging.info('Found %d new hyperparameter configurations.', len(new_hparam_configs))
        trainer.run_trainings(new_hparam_configs, args.steps)
        new_hparam_configs = config_reader.get_new_hparam_configs()
