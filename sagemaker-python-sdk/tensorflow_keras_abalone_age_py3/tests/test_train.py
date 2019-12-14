import tempfile
from unittest import TestCase

import os

from keras import Model

from main_train import train


class TestTrain(TestCase):
    def test_train(self):
        # Arrange
        train_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        train_file = "abalone_train.csv"
        val_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        val_file = "abalone_test.csv"

        model_dir = tempfile.mkdtemp()

        epochs = 2
        batch_size = 32

        # Act
        actual = train(train_dir, train_file, val_dir, val_file, model_dir, epochs, batch_size)

        # Assert
        self.assertIsInstance(actual, Model)
