import unittest
import lib
import tensorflow as tf


class TestLib(unittest.TestCase):

    def test_load_data(self):
        # Test if data loading function works correctly
        (x_train, y_train), (x_test, y_test) = lib.load_data()
        
        # Check if datasets are not empty
        self.assertTrue(len(x_train) > 0 and len(y_train) > 0)
        self.assertTrue(len(x_test) > 0 and len(y_test) > 0)

        # Check shapes of the datasets
        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))

    def test_create_model(self):
        # Test if model creation function works correctly
        model = lib.create_model()

        # Check if the model is of the correct type
        self.assertIsInstance(model, tf.keras.Sequential)

        # Check the model structure
        self.assertEqual(len(model.layers), 4)
        self.assertIsInstance(model.layers[0], tf.keras.layers.Flatten)
        self.assertIsInstance(model.layers[1], tf.keras.layers.Dense)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Dropout)
        self.assertIsInstance(model.layers[3], tf.keras.layers.Dense)

        # Check the output shape of the model
        self.assertEqual(model.output_shape, (None, 10))


if __name__ == '__main__':
    unittest.main()
