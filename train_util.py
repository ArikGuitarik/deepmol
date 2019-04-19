import os
import logging
import tensorflow as tf
import numpy as np
import time
import copy
from datetime import timedelta
from data.qm9_loader import QM9Loader
from data.molecules import TFMolBatch
from data.standardization import Standardization


class CurveSmoother:
    """This class imitates the behavior of the TensorBoard smoothing slider by performing a running average.

    For instance, to smooth a loss curve, instantiate the class and pass the values to the smooth function one by one.

    :param smoothing_factor: controls the amount of smoothing. 0 = no smoothing, 1 = stuck at initial value.
    :raises ValueError: If the smoothing factor is outside the valid range [0, 1].
    """

    def __init__(self, smoothing_factor):
        if not 0 <= smoothing_factor <= 1:
            raise ValueError('Smoothing factor must lie between 0 and 1.')
        self._smoothing_factor = smoothing_factor
        self._last_smoothed_value = None

    def smooth(self, new_value):
        """Smooth the value based on all previous values passed to this function.

        :param new_value: The latest value of the curve that is to be smoothed.
        :return: smoothed value
        """
        if self._last_smoothed_value is None:
            smoothed_value = new_value
        else:
            smoothed_value = self._last_smoothed_value * self._smoothing_factor + \
                             (1 - self._smoothing_factor) * new_value
        self._last_smoothed_value = smoothed_value
        return smoothed_value


class QM9Trainer:
    """Abstract base class for training and evaluating models on the QM9 data set.

    All the logic for loading data and performing training is defined here, while the model and the evaluation
    is to be defined in the concrete sub class.

    :param data_dir: directory containing the QM9 files *.sdf, *_labels.csv (*=[training|validation|test])
    :param train_log_interval: Write training log after this many steps.
    :param val_log_interval: Write validation log after this many steps.
    :param name: Name of the experiment/training that is performed.
    :param implicit_hydrogen: If True, hydrogen atoms will be treated implicitly.
    :param patience: Stop training early if the (smoothed) validation loss has not improved for this many steps.
    :param loss_smoothing: Early stopping is decided based on a running average of the validation loss.
        This parameter controls the amount of smoothing and corresponds to the TensorBoard smoothing slider.
    :param property_names: List of QM9 properties that should be used for training.
    """

    def __init__(self, data_dir, train_log_interval=250, val_log_interval=10000, name='', implicit_hydrogen=True,
                 patience=float('inf'), loss_smoothing=0.8, property_names=None):
        self.data_dir = data_dir
        self.name = name
        self.results_dir = os.path.join(os.path.dirname(__file__), name + '_results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.train_log_interval = train_log_interval
        self.val_log_interval = val_log_interval
        self.patience = patience
        self.loss_smoothing = loss_smoothing

        self.property_names = QM9Loader.all_property_names if property_names is None else property_names
        self.implicit_hydrogen = implicit_hydrogen

        # updated by run_trainings
        self._config_name = None  # name of currently trained hyperparameter configuration
        self._current_config_number = 0
        self._num_configs = 0

        # initialized by _prepare_data
        self._standardization = None
        self._train_iterator, self._val_iterator, self._test_iterator = None, None, None
        self._train_mols, self._val_mols, self._test_mols = None, None, None

        # initialized by _build_model
        self._global_step = None  # tf.Variable for controlling learning rate decay
        self._train_op = None

        # initialized by _train
        self._sess = None
        self._summary_writer = None

        # initialized by _init_saver and _restore_saved_model
        self._step = 0
        self._saver = None
        self._checkpoint_dir = None

    def run_trainings(self, hparam_configs, num_steps):
        """Run training and evaluation for different hyperparameter configurations.

        If a configuration with the same name has been trained before, its weights are restored from the checkpoint
        and training is started from the step saved in the checkpoint.
        If that step is higher than num_steps, training is skipped. To only run evaluation, e.g. use num_steps=0.

        :param hparam_configs: dict of tf.contrib.training.HParams objects
        :param num_steps: Number of steps (=batches) to train.
        """
        self._num_configs += len(hparam_configs)
        for config_name, hparam_config in hparam_configs.items():
            self._current_config_number += 1
            self._config_name = config_name
            logging.info('==== Configuration %d / %d: ' + config_name + ' ====', self._current_config_number,
                         self._num_configs)
            tf.reset_default_graph()
            self._sess = tf.Session()
            logging.info('Preparing data.')
            self._prepare_data(hparam_config.batch_size)
            logging.info('Building model.')
            self._build_model(hparam_config)
            self._init_saver()
            self._restore_saved_model()  # restore previously trained model from disk
            if num_steps > self._step:
                logging.info('Starting training.')
                self._train(num_steps)
                logging.info('Training complete.')
                self._restore_saved_model()  # restore model with best validation loss during current training
            logging.info('Starting evaluation.')
            self._eval_results()
            self._sess.close()
        logging.info('All trainings complete.')

    def _prepare_data(self, batch_size):
        """Create data iterators and TFMolBatches for the QM9 data.

        :param batch_size: Number of molecules per batch.
        """
        self._standardization = Standardization()
        self._train_iterator = self._create_data_iterator(batch_size, 'training', standardization=self._standardization)
        self._val_iterator = self._create_data_iterator(batch_size, 'validation', standardization=self._standardization)
        self._test_iterator = self._create_data_iterator(batch_size, 'test', standardization=self._standardization)

        with tf.name_scope('train_data'):
            train_data = self._train_iterator.get_next()
            self._train_mols = TFMolBatch(train_data['atoms'], mask=train_data['mask'], labels=train_data['labels'],
                                          distance_matrix=train_data['interactions'],
                                          coordinates=train_data['coordinates'])
        with tf.name_scope('val_data'):
            val_data = self._val_iterator.get_next()
            self._val_mols = TFMolBatch(val_data['atoms'], mask=val_data['mask'], labels=val_data['labels'],
                                        distance_matrix=val_data['interactions'], coordinates=val_data['coordinates'])
        with tf.name_scope('test_data'):
            test_data = self._test_iterator.get_next()
            self._test_mols = TFMolBatch(test_data['atoms'], mask=test_data['mask'], labels=test_data['labels'],
                                         distance_matrix=test_data['interactions'],
                                         coordinates=test_data['coordinates'])

    def _create_data_iterator(self, batch_size, partition='training', featurizer='distance', standardization=None):
        """Create a data iterator for one partition of the data set.

        :param batch_size: Number of molecules per batch.
        :param partition: [training|validation|test]
        :param featurizer: name of the featurizer to featurize the molecules.
        :param standardization: Standardization object, ensures that label standardization is the same in all partitions
        :return:
        """
        data_path = os.path.join(self.data_dir, partition + '.sdf')
        label_path = os.path.join(self.data_dir, partition + '_labels.csv')
        qm9_loader = QM9Loader(data_path, label_path, implicit_hydrogen=self.implicit_hydrogen,
                               property_names=self.property_names, label_standardization=standardization,
                               featurizer=featurizer)
        data_set = qm9_loader.create_tf_dataset()
        data_set = data_set.cache()
        if partition == 'training':
            data_set = data_set.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            data_set = data_set.repeat()
        data_set = data_set.batch(batch_size)
        data_set = data_set.prefetch(buffer_size=1)
        if partition == 'training':
            return data_set.make_one_shot_iterator()
        return data_set.make_initializable_iterator()

    def _init_saver(self):
        """Initialize the tf.train.Saver to save and restore the model to/from disk."""
        self._saver = tf.train.Saver(max_to_keep=1)
        self._checkpoint_dir = os.path.join(self.results_dir, 'checkpoints', 'checkpoints_' + self._config_name)
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

    def _restore_saved_model(self):
        """Restore the model from checkpoint (if available) and set the current training step accordingly. """
        latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
        if latest_checkpoint is None:
            self._step = 0
        else:
            self._saver.restore(self._sess, latest_checkpoint)
            self._step = int(latest_checkpoint.split('-')[-1])

    def _train(self, num_steps):
        """Perform training until num_steps is reached or the validation loss stops improving (early stopping).

        Training starts at self._step with an initial validation via self._write_val_log().
        At each step, self._train_op is run.
        Every self.train_log_interval steps, self._write_train_log() is called.
        Every self.val_log_interval steps, self._write_val_log() is called.
        If the validation loss has improved, the model is saved to disk.
        Early stopping is done when the smoothed validation loss curve has not improved for self.patience steps.

        :param num_steps: Training step at which training stops.
        """
        sess = self._sess
        self._summary_writer = tf.summary.FileWriter(
            os.path.join(self.results_dir, 'logs', 'logs_' + self._config_name), sess.graph)
        sess.run(tf.global_variables_initializer())
        start_step = self._step
        sess.run(self._global_step.assign(start_step))
        sess.graph.finalize()

        early_stopping_smoother = CurveSmoother(smoothing_factor=self.loss_smoothing)
        best_val_loss = self._write_val_log()
        best_val_loss_smoothed = early_stopping_smoother.smooth(best_val_loss)
        best_step_smoothed = start_step
        logging.info('%d / %d: Initial validation yields loss %f', self._step, num_steps, best_val_loss)

        start_time = time.time()
        for self._step in range(start_step + 1, num_steps + 1):
            # check early stopping
            if (self._step - best_step_smoothed) > self.patience:
                logging.info('Out of patience.')
                break

            # validate
            if self._step % self.val_log_interval == 0:
                val_loss = self._write_val_log()
                best_log = ''
                if val_loss < best_val_loss:
                    self._saver.save(sess, os.path.join(self._checkpoint_dir, 'checkpoint'), self._step)
                    best_val_loss = val_loss
                    best_log = ' (BEST so far)'
                logging.info('%d / %d: Validation yields loss %f' + best_log, self._step, num_steps, val_loss)

                # smooth validation loss for early stopping
                val_loss_smoothed = early_stopping_smoother.smooth(val_loss)
                if val_loss_smoothed < best_val_loss_smoothed:
                    best_val_loss_smoothed = val_loss_smoothed
                    best_step_smoothed = self._step

            # train
            if self._step % self.train_log_interval == 0:
                self._write_train_log()
                logging.info('%d / %d: Training summary written', self._step, num_steps)
            else:
                sess.run(self._train_op)

            # estimate remaining time
            if self._step % (10 * self.train_log_interval) == 0 and (self._step - start_step) >= self.val_log_interval:
                seconds_since_start = time.time() - start_time
                remaining_seconds = seconds_since_start * (num_steps / self._step - 1)
                formatted_remaining_time = str(timedelta(seconds=int(remaining_seconds)))
                logging.info(formatted_remaining_time + ' remaining for configuration %d / %d',
                             self._current_config_number, self._num_configs)

    def _build_model(self, hparams):
        """Build the model given the hyperparameter configuration.

        Needs to be implemented in subclass and initialize self._train_op and self._global_step.

        :param hparams: tf.contrib.training.HParams object
        """
        raise NotImplementedError('Model must be defined in child class.')

    def _write_train_log(self):
        """Perform training step and and write training log. Needs to be implemented in subclass."""
        raise NotImplementedError('Must be implemented in child class.')

    def _write_val_log(self):
        """Perform validation, write validation log and return validation loss. Needs to be implemented in subclass."""
        raise NotImplementedError('Must be implemented in child class.')

    def _eval_results(self):
        """Evaluate results after training is complete. Needs to be implemented in subclass."""
        raise NotImplementedError('Must be implemented in child class.')

    def _write_eval_results_to_file(self, result_dict):
        """Write results of post-training evaluation into one central results file.

        :param result_dict: Dictionary containing evaluation results.
        """
        results_filename = os.path.join(self.results_dir, 'results.txt')
        if not os.path.isfile(results_filename):  # file does not exist yet
            with open(results_filename, 'w') as f:
                header = 'config' + '\t' + '\t'.join(result_dict.keys()) + '\n'
                f.write(header)
        with open(results_filename, 'a') as f:
            data = self._config_name + '\t' + '\t'.join([str(v) for v in result_dict.values()]) + '\n'
            f.write(data)
        logging.info('Evaluation results for config ' + self._config_name + ' written to ' + results_filename)

    def _average_over_dataset(self, data_iterator, eval_tensors):
        """Calculate the average values of eval_tensors across the specified data set.

        :param data_iterator: The initializable_iterator of the relevant data set.
        :param eval_tensors: The one-dimensional tensors to be evaluated and averaged.
        :return: The average values of eval_tensors.
        """
        self._sess.run(data_iterator.initializer)
        values = []
        while True:
            try:
                value = self._sess.run(eval_tensors)
                values.append(value)
            except tf.errors.OutOfRangeError:
                break
        values_np = np.array(values)
        avg_values = np.mean(values_np, axis=0)

        return avg_values


class ConfigReader:
    """Read hyperparameter configurations from json files.

    Keeps track of whether new files have been added since the last read, such that after training completion,
    training can continue directly with new configurations that have been added in the meantime.

    :param config_dir: All files in this directory ending with .json will be read.
    :param default_hparams: The tf.contrib.training.HParams object with the default hyperparameter values.
    """

    def __init__(self, config_dir, default_hparams):
        self.config_dir = config_dir
        self.previous_config_files = set()
        self.default_hparams = default_hparams

    def _get_new_config_files(self):
        """List all json files in config_dir that have not been there at the previous call to this method.

        :return: List of file paths.
        """
        config_files = {os.path.join(self.config_dir, f) for f in os.listdir(self.config_dir) if f.endswith('.json')}
        new_config_files = config_files - self.previous_config_files
        self.previous_config_files |= new_config_files
        return new_config_files

    def get_new_hparam_configs(self):
        """Get all hyperparameter configs in config_dir that have not been there at the previous call to this method.

        :return: dict of hyperparameter configs; config_name => tf.contrib.training.HParams object
        """
        new_config_files = self._get_new_config_files()
        hparam_configs = {}
        for config_file in new_config_files:
            with open(config_file, 'r') as f:
                hparams = copy.deepcopy(self.default_hparams)
                filename = os.path.basename(config_file)
                config_name = os.path.splitext(filename)[0]
                hparam_configs[config_name] = hparams.parse_json(f.read())

        return hparam_configs
