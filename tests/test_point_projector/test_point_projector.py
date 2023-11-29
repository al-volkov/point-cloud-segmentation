import unittest

import numpy as np

from src.point_projector.point_projector import PointProjector


class TestPointProjector(unittest.TestCase):
    def setUp(self):
        self.image_width = 640
        self.image_height = 480
        self.vertical_offset = 0
        self.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.camera_coordinates = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        self.camera_angles = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])

    def test_vectorized_true(self):
        point_projector = PointProjector(
            self.image_width,
            self.image_height,
            self.vertical_offset,
            self.points,
            self.camera_coordinates,
            self.camera_angles,
            vectorized=True,
        )
        self.assertIsInstance(point_projector.projected_points, np.ndarray)

    def test_vectorized_false(self):
        point_projector = PointProjector(
            self.image_width,
            self.image_height,
            self.vertical_offset,
            self.points,
            self.camera_coordinates,
            self.camera_angles,
            vectorized=False,
        )
        projected_points = point_projector.project_all(
            self.points, self.camera_coordinates, self.camera_angles
        )
        self.assertIsInstance(projected_points, np.ndarray)


if __name__ == "__main__":
    unittest.main()
