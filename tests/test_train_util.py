import unittest
import numpy.testing as npt
from train_util import CurveSmoother


class TestCurveSmoother(unittest.TestCase):
    def test_smoothing_factor_range(self):
        with self.assertRaises(ValueError):
            CurveSmoother(-12)
        with self.assertRaises(ValueError):
            CurveSmoother(1.1)
        CurveSmoother(0.4)

    def test_smoothing(self):
        data = [-1, 2, 4, 10]
        smoothed_data_ref = [-1, 0.2, 1.72, 5.032]

        smoother = CurveSmoother(0.6)
        smoothed_data = [smoother.smooth(val) for val in data]
        npt.assert_allclose(smoothed_data, smoothed_data_ref)

        smoother = CurveSmoother(0)
        smoothed_data = [smoother.smooth(val) for val in data]
        npt.assert_allclose(smoothed_data, data)

        smoother = CurveSmoother(1)
        smoothed_data = [smoother.smooth(val) for val in data]
        npt.assert_allclose(smoothed_data, [data[0] for _ in data])


if __name__ == '__main__':
    unittest.main()
