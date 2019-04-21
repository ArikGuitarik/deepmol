import unittest
import numpy.testing as npt
import os
from train_util import CurveSmoother, ConfigReader
import tensorflow as tf


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


class TestConfigReader(unittest.TestCase):
    def test_reader(self):
        hparams_1 = tf.contrib.training.HParams(x=2, y=3, z=True)
        hparams_2 = tf.contrib.training.HParams(x=2.3, y=3, z=True)
        hparams_3 = tf.contrib.training.HParams(x=3.14, y=42, z=False)
        hparams_default = tf.contrib.training.HParams(x=1.8, y=2.9, z=False)

        tmpdir = 'unittest_configs'
        tmpfile_1 = os.path.join(tmpdir, '1.json')
        tmpfile_2 = os.path.join(tmpdir, '2.json')
        tmpfile_3 = os.path.join(tmpdir, '3.json')
        try:
            os.makedirs(tmpdir, exist_ok=True)
            with open(tmpfile_1, 'w') as f:
                f.write(hparams_1.to_json())
            with open(tmpfile_2, 'w') as f:
                f.write(hparams_2.to_json())

            config_reader = ConfigReader(tmpdir, hparams_default)
            new_configs = config_reader.get_new_hparam_configs()
            self.assertEqual(len(new_configs), 2)
            self.assertDictEqual(hparams_1.values(), new_configs['1'].values())
            self.assertDictEqual(hparams_2.values(), new_configs['2'].values())

            new_configs = config_reader.get_new_hparam_configs()
            self.assertEqual(len(new_configs), 0)

            os.remove(tmpfile_1)

            with open(tmpfile_3, 'w') as f:
                f.write(hparams_3.to_json())

            new_configs = config_reader.get_new_hparam_configs()
            self.assertEqual(len(new_configs), 1)
            self.assertDictEqual(hparams_3.values(), new_configs['3'].values())
        finally:
            os.remove(tmpfile_2)
            os.remove(tmpfile_3)
            os.rmdir(tmpdir)


if __name__ == '__main__':
    unittest.main()
