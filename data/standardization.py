import numpy as np


class Standardization:
    """Define a standardization operation that can be applied to different data sets.

    We would like to standardize our training data to zero mean and unit variance. This transformation defines the
    scales of our data. After standardizing the training set, every further call of apply() will scale the data
    accordingly by applying the same transformation as the first time.
    The "data" must be an ndarray where the first dimension runs over the different data points.
    """

    def __init__(self):
        self._means = None
        self._standard_deviations = None

    def is_defined(self):
        """Has the standardization operation been defined yet?"""
        return (self._means is not None) and (self._standard_deviations is not None)

    def define(self, reference_data):
        """Define the standardization such that reference_data would be transformed to zero mean, unit variance."""
        self._means = np.mean(reference_data, axis=0)
        self._standard_deviations = np.std(reference_data, axis=0)

    def apply(self, input_data):
        """Apply standardization using previous definition or, if undefined, the input_data."""
        if not self.is_defined():
            self.define(input_data)

        return (input_data - self._means) / self._standard_deviations

    def undo(self, standardized_data):
        """Convert data back to its original scales."""
        if not self.is_defined():
            raise TypeError('This Standardizer has not been initialized yet.')

        return standardized_data * self._standard_deviations + self._means
