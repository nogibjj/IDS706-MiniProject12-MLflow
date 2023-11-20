import unittest
import lib
import tensorflow as tf


class TestLib(unittest.TestCase):

    def test_load_data(self):
        # Test if data loading function with TFDS works correctly
        ds_train, ds_test = lib.load_data()

        # Check if datasets are not empty
        self.assertTrue(isinstance(ds_train, tf.data.Dataset))
        self.assertTrue(isinstance(ds_test, tf.data.Dataset))

        # Check one batch of data
        for images, labels in ds_train.take(1):
            self.assertEqual(images.shape, (128, 28, 28, 1))
            self.assertEqual(labels.shape, (128,))
            self.assertTrue((images.numpy() >= 0).all() and (images.numpy() <= 1).all())


if __name__ == '__main__':
    unittest.main()
