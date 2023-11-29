import unittest
import numpy as np
from unittest.mock import Mock
from src.point_cloud_segmentor import PointCloudSegmentor

class TestPointCloudSegmentor(unittest.TestCase):
    def setUp(self):
        self.point_cloud = Mock()
        self.point_cloud.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.image_loader = Mock()
        self.image_width = 640
        self.image_height = 480
        self.vertical_offset = 0
        self.camera_coordinates = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        self.camera_angles = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
        self.predictor = Mock()
        self.closest_images = np.array([0, 1, 2])
        self.true_labels = np.array([0, 1, 2])
        self.segmentor = PointCloudSegmentor(
            self.point_cloud,
            self.camera_coordinates,
            self.camera_angles,
            self.image_loader,
            self.image_width,
            self.image_height,
            self.vertical_offset,
            self.predictor,
            self.closest_images,
            self.true_labels,
        )

    def test_init(self):
        self.assertEqual(self.segmentor.point_cloud, self.point_cloud)
        self.assertEqual(self.segmentor.image_loader, self.image_loader)
        self.assertEqual(self.segmentor.image_width, self.image_width)
        self.assertEqual(self.segmentor.image_height, self.image_height)
        self.assertEqual(self.segmentor.vertical_offset, self.vertical_offset)
        self.assertTrue((self.segmentor.camera_coordinates == self.camera_coordinates).all())
        self.assertTrue((self.segmentor.camera_angles == self.camera_angles).all())
        self.assertEqual(self.segmentor.predictor, self.predictor)
        self.assertTrue((self.segmentor.closest_images == self.closest_images).all())
        self.assertTrue((self.segmentor.true_labels == self.true_labels).all())

    def test_evaluate(self):
        predicted_labels = np.array([0, 1, 2])
        self.segmentor.evaluate(predicted_labels)

if __name__ == "__main__":
    unittest.main()