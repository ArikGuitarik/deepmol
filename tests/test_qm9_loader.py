import os
from data.qm9_loader import QM9Loader
import tensorflow as tf
import numpy as np
from data.featurizer import DistanceFeaturizer
import numpy.testing as npt


class TestQM9Loader(tf.test.TestCase):
    def test_data_set_generation(self):
        data_import_directory = os.path.dirname(__file__)
        sdf_path = os.path.join(data_import_directory, '100_sample_molecules.sdf')
        label_path = os.path.join(data_import_directory, '100_sample_molecules_labels.csv')

        properties = ['mu', 'alpha', 'Cv']
        implicit_hydrogen = True
        qm9_loader = QM9Loader(sdf_path, label_path, implicit_hydrogen=implicit_hydrogen, property_names=properties)
        batch_size = 16
        n = 9 if implicit_hydrogen else 29
        data_set = qm9_loader.create_tf_dataset().batch(batch_size)
        mol_batch = data_set.make_one_shot_iterator().get_next()

        with self.test_session() as sess:
            for i in range(4):
                data = sess.run(mol_batch)
                npt.assert_equal(data['atoms'].shape,
                                 np.array([batch_size, n, DistanceFeaturizer.num_atom_features]))
                npt.assert_equal(data['interactions'].shape,
                                 np.array([batch_size, n, n, DistanceFeaturizer.num_interaction_features]))
                npt.assert_equal(data['mask'].shape, np.array([batch_size, n]))
                npt.assert_equal(data['labels'].shape, np.array([batch_size, len(properties)]))


if __name__ == '__main__':
    tf.test.main()
