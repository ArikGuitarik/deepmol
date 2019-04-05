import unittest
import numpy as np
import numpy.testing as npt
from data.standardization import Standardization


class TestStandardization(unittest.TestCase):
    example_data = np.arange(12).reshape(3, 4) ** 2
    example_data_2 = np.arange(1, 13).reshape(3, 4) ** 2

    def test_initialization(self):
        standardization = Standardization()
        self.assertFalse(standardization.is_defined())
        with self.assertRaises(TypeError):
            standardization.undo(self.example_data)

    def test_apply(self):
        standardization = Standardization()
        standardized_data = standardization.apply(self.example_data)
        actual_variance = np.var(standardized_data, axis=0)
        desired_variance = np.ones_like(actual_variance)
        npt.assert_allclose(actual_variance, desired_variance)

        actual_mean = np.mean(standardized_data, axis=0)
        desired_mean = np.zeros_like(actual_mean)
        npt.assert_allclose(actual_mean, desired_mean, atol=1e-8)

    def test_undo(self):
        standardization = Standardization()
        standardized_data = standardization.apply(self.example_data)
        retrieved_data = standardization.undo(standardized_data)
        npt.assert_allclose(retrieved_data, self.example_data)

    def test_repeated_application(self):
        standardization = Standardization()
        standardization.apply(self.example_data)
        standardized_data_2 = standardization.apply(self.example_data_2)
        desired = (self.example_data_2 - np.mean(self.example_data, axis=0)) / np.std(self.example_data, axis=0)
        npt.assert_allclose(standardized_data_2, desired)


if __name__ == '__main__':
    unittest.main()
