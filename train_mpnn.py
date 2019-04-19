import logging
import tensorflow as tf
from train_util import QM9Trainer
from model.mpnn import MPNN

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


class MPNNTrainer(QM9Trainer):
    """Extend QM9Trainer to train and evaluate a message passing neural network for property prediction.

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
    def __init__(self, data_dir, train_log_interval, val_log_interval, name='', implicit_hydrogen=True,
                 patience=float('inf'), loss_smoothing=0.8, property_names=None):
        super(MPNNTrainer, self).__init__(data_dir, train_log_interval, val_log_interval, name, implicit_hydrogen,
                                          patience, loss_smoothing, property_names)
        # initialized by _build_model
        self._train_summary = None
        self._val_loss = None
        self._val_mae_actual_scale = None
        self._test_loss = None
        self._test_mae_actual_scale = None

    def _write_train_log(self):
        """Perform training step and and write training log. Overrides superclass method."""
        summary, _ = self._sess.run([self._train_summary, self._train_op])
        self._summary_writer.add_summary(summary, self._step)

    def _write_val_log(self):
        """Perform validation, write validation log and return validation loss. Overrides superclass method.

        :return: validation loss
        """
        averages = self._average_over_dataset(self._val_iterator, [self._val_loss, self._val_mae_actual_scale])
        val_loss, val_mae = averages[0], averages[1]

        summary = tf.Summary()
        summary.value.add(tag='avg_val_loss', simple_value=val_loss)
        summary.value.add(tag='avg_val_mae', simple_value=val_mae)
        self._summary_writer.add_summary(summary, self._step)
        self._summary_writer.flush()

        return val_loss

    def _eval_results(self):
        pass  # todo

    def _build_model(self, hparams):
        """Build the model given the hyperparameter configuration. Overrides superclass method.

        :param hparams: tf.contrib.training.HParams object
        """
        mpnn = MPNN(hparams, output_dim=len(self.property_names))
        train_output = mpnn.forward(self._train_mols)
        train_loss = tf.losses.mean_squared_error(labels=self._train_mols.labels, predictions=train_output)

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(hparams.learning_rate, self._global_step, hparams.lr_decay_steps,
                                                   hparams.lr_decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train_op = optimizer.minimize(train_loss, global_step=self._global_step)

        train_labels_actual_scale = self._standardization.undo(self._train_mols.labels)
        train_output_actual_scale = self._standardization.undo(train_output)
        train_mae_actual_scale = tf.losses.absolute_difference(train_labels_actual_scale, train_output_actual_scale)
        self._train_summary = tf.summary.merge([tf.summary.scalar('train_loss', train_loss),
                                                tf.summary.scalar('train_mae_actual_scale', train_mae_actual_scale)])

        # validation summary
        val_output = mpnn.forward(self._val_mols)
        self._val_loss = tf.losses.mean_squared_error(labels=self._val_mols.labels, predictions=val_output)

        val_labels_actual_scale = self._standardization.undo(self._val_mols.labels)
        val_output_actual_scale = self._standardization.undo(val_output)
        self._val_mae_actual_scale = tf.losses.absolute_difference(val_labels_actual_scale, val_output_actual_scale)

        # test summary
        test_output = mpnn.forward(self._test_mols)
        self._test_loss = tf.losses.mean_squared_error(labels=self._test_mols.labels, predictions=test_output)

        test_labels_actual_scale = self._standardization.undo(self._test_mols.labels)
        test_output_actual_scale = self._standardization.undo(test_output)
        self._test_mae_actual_scale = tf.losses.absolute_difference(test_labels_actual_scale, test_output_actual_scale)
