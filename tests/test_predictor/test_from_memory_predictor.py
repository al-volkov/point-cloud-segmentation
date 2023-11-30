import os
import unittest
from unittest.mock import Mock

import numpy as np

from src.predictor.from_memory_predictor import FromMemoryPredictor


class TestFromMemoryPredictor(unittest.TestCase):
    def setUp(self):
        self.image_loader = Mock()
        self.predictions_path = os.path.join("tests", "data", "predictions.npy")
        self.predictions = np.random.rand(5, 3, 3)
        np.save(self.predictions_path, self.predictions.transpose((0, 2, 1)))
        self.predictor = FromMemoryPredictor(self.image_loader, self.predictions_path)

    def test_init(self):
        self.assertEqual(self.predictor.image_loader, self.image_loader)
        self.assertTrue((self.predictor.predictions == self.predictions).all())

    def test_get_classes(self):
        classes = self.predictor.get_classes()
        expected_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.assertEqual(classes, expected_classes)

    def test_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            FromMemoryPredictor(self.image_loader, "non_existent_file.npy")

    def test_predict(self):
        index = 0
        prediction = self.predictor.predict(index)
        self.assertTrue((prediction == self.predictions[index]).all())

    def test_predict_multiple(self):
        indices = np.array([0, 1, 2])
        predictions = self.predictor.predict_multiple(indices)
        self.assertTrue((predictions == self.predictions[indices]).all())

    def tearDown(self):
        import os

        os.remove(self.predictions_path)


if __name__ == "__main__":
    unittest.main()
