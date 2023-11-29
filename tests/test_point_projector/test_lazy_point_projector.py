import unittest

import numpy as np

from src.point_projector.lazy_point_projector import LazyPointProjector


class TestLazyPointProjector(unittest.TestCase):
    def setUp(self):
        self.image_width = 640
        self.image_height = 480
        self.vertical_offset = 0
        self.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.camera_coordinates = np.array([[0, 2, 3], [2, 2, 2], [3, 3, 3]])
        self.camera_angles = np.array([[0, -90, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
        self.lazy_point_projector = LazyPointProjector(
            self.image_width,
            self.image_height,
            self.vertical_offset,
            self.points,
            self.camera_coordinates,
            self.camera_angles,
        )

    def test_init(self):
        np.testing.assert_array_equal(self.lazy_point_projector.points, self.points)
        np.testing.assert_array_equal(
            self.lazy_point_projector.camera_coordinates, self.camera_coordinates
        )
        np.testing.assert_array_equal(
            self.lazy_point_projector.camera_angles, self.camera_angles
        )


if __name__ == "__main__":
    unittest.main()
