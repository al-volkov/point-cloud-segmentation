import math as m
from typing import Optional

import numpy as np


class PointProjectorCore:
    absolute_width = 8000
    absolute_height = 4000
    x_scale = 0.9997720449
    y_scale = 0.9997720449
    z_scale = 1
    scale = np.array([x_scale, y_scale, z_scale])

    def __init__(
        self,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        apply_scale=False,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_offset = vertical_offset
        self.apply_scale = apply_scale
        self.start_x = (self.absolute_width - image_width) // 2
        self.start_y = (self.absolute_height - image_height) // 2 + vertical_offset

    def project_on_image(
        self,
        point_coordinates: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> Optional[np.ndarray]:
        coordinates_on_original_image = self._project_on_original_image(
            point_coordinates, camera_coordinates, camera_angles
        )
        return self.convert_coordinates(coordinates_on_original_image)

    def convert_coordinates(
        self, original_coordinates: np.ndarray
    ) -> Optional[np.ndarray]:
        new_x = original_coordinates[0] - self.start_x
        new_y = original_coordinates[1] - self.start_y
        if 0 <= new_x < self.image_width and 0 <= new_y < self.image_height:
            return np.array([new_x, new_y]).astype(np.int16)
        return None

    def _project_on_original_image(
        self,
        point_coordinates: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> np.ndarray:
        adjusted_angles = camera_angles.copy()
        adjusted_angles[2] += 90  # rotate view by 90 degrees
        vector_coordinates = camera_coordinates - point_coordinates
        if self.apply_scale:
            vector_coordinates = np.multiply(vector_coordinates, self.scale)
        rotated_vector_coordinates = self.rotate_vector(
            vector_coordinates, adjusted_angles
        )
        spherical_coordinates = self.convert_to_spherical_coordinates(
            rotated_vector_coordinates
        )
        r, theta, phi = spherical_coordinates
        coordinates_on_image = self.spherical_to_equirectangular(theta, phi)  # x, y
        return coordinates_on_image

    def rotate_vector(
        self, vector_coordinates: np.ndarray, camera_angles: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            self.get_rotation_matrix(camera_angles) @ vector_coordinates
        ).reshape(-1)

    def get_rotation_matrix(self, camera_angles: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = camera_angles
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        R_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        R_pitch = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        R_yaw = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        R = R_yaw @ R_pitch @ R_roll
        return R

    def convert_to_spherical_coordinates(
        self, vector_coordinates: np.ndarray
    ) -> np.ndarray:
        x, y, z = vector_coordinates
        r = m.sqrt(x**2 + y**2 + z**2)
        theta = m.atan2(y, x)
        phi = m.acos(z / r) if r != 0 else 0
        return np.array([r, theta, phi])

    def spherical_to_equirectangular(
        self, theta: np.float64, phi: np.float64
    ) -> np.ndarray:
        x = (-theta + m.pi) * self.absolute_width / (2 * m.pi)
        y = (m.pi - phi) * self.absolute_height / m.pi
        return np.array([round(x), round(y)])
