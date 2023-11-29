import numpy as np


class PointProjectorCoreVectorized:
    absolute_width = 8000
    absolute_height = 4000
    x_scale = 0.9997720449
    y_scale = 0.9997720449
    z_scale = 1
    scale = np.array([x_scale, y_scale, z_scale])

    __slots__ = (
        "image_width",
        "image_height",
        "vertical_offset",
        "apply_scale",
        "start_x",
        "start_y",
    )

    def __init__(
        self,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        apply_scale=False,
    ):
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
    ) -> np.ndarray:
        coordinates_on_original_image = self._project_on_original_image(
            point_coordinates, camera_coordinates, camera_angles
        )
        return self.convert_coordinates(coordinates_on_original_image)

    def convert_coordinates(self, original_coordinates: np.ndarray) -> np.ndarray:
        new_x = original_coordinates[..., 0] - self.start_x
        new_y = original_coordinates[..., 1] - self.start_y

        valid_x = (0 <= new_x) & (new_x < self.image_width)
        valid_y = (0 <= new_y) & (new_y < self.image_height)
        valid_mask = valid_x & valid_y

        converted_coordinates = np.full(original_coordinates.shape, -1, dtype=np.int16)

        converted_coordinates[..., 0][valid_mask] = new_x[valid_mask]
        converted_coordinates[..., 1][valid_mask] = new_y[valid_mask]

        return converted_coordinates

    def _project_on_original_image(
        self,
        point_coordinates: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> np.ndarray:
        point_coordinates = point_coordinates[:, np.newaxis, :]
        camera_coordinates = camera_coordinates[np.newaxis, :, :]
        adjusted_camera_angles = camera_angles.copy()
        adjusted_camera_angles[:, 2] += 90  # rotate view by 90 degrees
        adjusted_camera_angles = adjusted_camera_angles[np.newaxis, :, :]
        vector_coordinates = camera_coordinates - point_coordinates
        if self.apply_scale:
            vector_coordinates *= self.scale
        rotated_vector_coordinates = self.rotate_vector(
            vector_coordinates, adjusted_camera_angles
        )
        spherical_coordinates = self.convert_to_spherical_coordinates(
            rotated_vector_coordinates
        )
        theta = spherical_coordinates[..., 1]
        phi = spherical_coordinates[..., 2]
        coordinates_on_image = self.spherical_to_equirectangular(theta, phi)  # x, y
        return coordinates_on_image

    def rotate_vector(
        self, vector_coordinates: np.ndarray, camera_angles: np.ndarray
    ) -> np.ndarray:
        rotation_matrices = self.get_rotation_matrix(camera_angles)
        rotated_vectors = np.einsum(
            "ijl,jkl->ijk", vector_coordinates, rotation_matrices
        )
        return rotated_vectors

    def get_rotation_matrix(self, camera_angles: np.ndarray) -> np.ndarray:
        camera_angles = np.radians(camera_angles)
        roll = camera_angles[0, :, 0]
        pitch = camera_angles[0, :, 1]
        yaw = camera_angles[0, :, 2]
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        zeros = np.zeros_like(roll)
        ones = np.ones_like(roll)

        R_roll = np.array(
            [
                [ones, zeros, zeros],
                [zeros, cos_roll, -sin_roll],
                [zeros, sin_roll, cos_roll],
            ]
        ).transpose((2, 0, 1))  # Transposed to shape (m, 3, 3)

        R_pitch = np.array(
            [
                [cos_pitch, zeros, sin_pitch],
                [zeros, ones, zeros],
                [-sin_pitch, zeros, cos_pitch],
            ]
        ).transpose((2, 0, 1))  # Transposed to shape (m, 3, 3)

        R_yaw = np.array(
            [
                [cos_yaw, -sin_yaw, zeros],
                [sin_yaw, cos_yaw, zeros],
                [zeros, zeros, ones],
            ]
        ).transpose((2, 0, 1))

        # Combine the rotations
        R = R_yaw @ R_pitch @ R_roll
        # print(f"Rotation matrix shape = {R.shape}")
        return R

    def convert_to_spherical_coordinates(
        self, vector_coordinates: np.ndarray
    ) -> np.ndarray:
        x = vector_coordinates[..., 0]
        y = vector_coordinates[..., 1]
        z = vector_coordinates[..., 2]

        r = np.sqrt(x**2 + y**2 + z**2)

        theta = np.arctan2(y, x)

        phi = np.arccos(np.clip(z / r, -1, 1))

        return np.stack((r, theta, phi), axis=-1)

    def spherical_to_equirectangular(
        self, theta: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        x = (-theta + np.pi) * self.absolute_width / (2 * np.pi)
        y = (np.pi - phi) * self.absolute_height / np.pi

        x = np.round(x).astype(np.int16)
        y = np.round(y).astype(np.int16)

        coordinates = np.stack((x, y), axis=-1)
        return coordinates
