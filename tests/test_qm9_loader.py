import os
from data.qm9_loader import QM9Loader
import tensorflow as tf
import numpy as np
from data.featurizer import DistanceNumHFeaturizer
import numpy.testing as npt


class TestQM9Loader(tf.test.TestCase):
    def test_data_set_generation(self):

        properties = ['mu', 'alpha', 'Cv']
        implicit_hydrogen = True
        n = 9 if implicit_hydrogen else 29

        featurizer = DistanceNumHFeaturizer(n, implicit_hydrogen)
        data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
        qm9_loader = QM9Loader(data_dir, featurizer, property_names=properties, standardize_labels=True)
        batch_size = 16
        data_set = qm9_loader.train_data.batch(batch_size)
        mol_batch = data_set.make_one_shot_iterator().get_next()

        with self.test_session() as sess:
            for i in range(4):
                data = sess.run(mol_batch)
                npt.assert_equal(data['atoms'].shape,
                                 np.array([batch_size, n, featurizer.num_atom_features]))
                npt.assert_equal(data['interactions'].shape,
                                 np.array([batch_size, n, n, featurizer.num_interaction_features]))
                npt.assert_equal(data['labels'].shape, np.array([batch_size, len(properties)]))


if __name__ == '__main__':
    tf.test.main()
