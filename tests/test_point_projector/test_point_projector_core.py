import unittest

import numpy as np

from src.point_projector.point_projector_core import PointProjectorCore


class TestPointProjectorCore(unittest.TestCase):
    def setUp(self):
        self.image_width = 640
        self.image_height = 480
        self.vertical_offset = 0
        self.apply_scale = False
        self.point_projector = PointProjectorCore(
            self.image_width, self.image_height, self.vertical_offset, self.apply_scale
        )
        self.point_coordinates = np.array([1, 2, 3])
        self.camera_coordinates = np.array([1, 1, 1])
        self.camera_angles = np.array([0, 0, 0])

    def test_init(self):
        self.assertEqual(self.point_projector.image_width, self.image_width)
        self.assertEqual(self.point_projector.image_height, self.image_height)
        self.assertEqual(self.point_projector.vertical_offset, self.vertical_offset)
        self.assertEqual(self.point_projector.apply_scale, self.apply_scale)

    def test_project_on_original_image(self):
        projected_points = self.point_projector._project_on_original_image(
            self.point_coordinates, self.camera_coordinates, self.camera_angles
        )
        self.assertIsInstance(projected_points, np.ndarray)

    def test_rotate_vector(self):
        vector_coordinates = np.array([1, 2, 3])
        rotated_vectors = self.point_projector.rotate_vector(
            vector_coordinates, self.camera_angles
        )
        self.assertIsInstance(rotated_vectors, np.ndarray)

    def test_get_rotation_matrix(self):
        rotation_matrices = self.point_projector.get_rotation_matrix(self.camera_angles)
        self.assertIsInstance(rotation_matrices, np.ndarray)

    def test_convert_to_spherical_coordinates(self):
        vector_coordinates = np.array([1, 2, 3])
        spherical_coordinates = self.point_projector.convert_to_spherical_coordinates(
            vector_coordinates
        )
        self.assertIsInstance(spherical_coordinates, np.ndarray)

    def test_spherical_to_equirectangular(self):
        theta = 0.5
        phi = 0.5
        coordinates = self.point_projector.spherical_to_equirectangular(theta, phi)
        self.assertIsInstance(coordinates, np.ndarray)


if __name__ == "__main__":
    unittest.main()
