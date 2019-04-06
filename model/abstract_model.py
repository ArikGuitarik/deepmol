import tensorflow as tf


class Model:
    """Abstract superclass for reusable models using weight sharing with tf.make_template.
    Subclasses implementing the _forward() method can then be called using .forward().

    Args:
        hparams: hyperparameters, as a tf.contrib.training.HParams object
    """
    def __init__(self, hparams):
        scope_name = self.__class__.__name__
        self.hparams = hparams
        self.forward = tf.make_template(scope_name, self._forward)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("Model subclasses must define _forward() method")
