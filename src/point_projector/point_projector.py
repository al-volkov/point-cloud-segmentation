from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from src.point_projector.point_projector_core import PointProjectorCore
from src.point_projector.point_projector_core_vectorized import (
    PointProjectorCoreVectorized,
)


class PointProjector:
    """
    Class for projecting 3D points onto 2D images.

    Args:
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        vertical_offset (int): The vertical offset of the image.
        points (np.ndarray): The array of 3D points to be projected.
        camera_coordinates (np.ndarray): The array of camera coordinates.
        camera_angles (np.ndarray): The array of camera angles.
        vectorized (bool, optional):\
            Whether to use vectorized projection. Defaults to True.

    Attributes:
        projector (Union[PointProjectorCore, PointProjectorCoreVectorized]):\
            The point projector core.
        points (np.ndarray): The array of 3D points.
        camera_coordinates (np.ndarray): The array of camera coordinates.
        camera_angles (np.ndarray): The array of camera angles.
        projected_points (np.ndarray): The array of projected points.
        valid_projection_mask (np.ndarray): The mask indicating valid projections.

    Methods:
        project_all: Projects all points onto the images.
        project: Projects a specific point onto a specific image.
    """

    __slots__ = (
        "projector",
        "points",
        "camera_coordinates",
        "camera_angles",
        "projected_points",
        "valid_projection_mask",
    )

    def __init__(
        self,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        points: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
        vectorized=True,
    ) -> None:
        """
        Initializes the PointProjector object.

        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image.
            vertical_offset (int): The vertical offset of the image.
            points (np.ndarray): The array of 3D points to be projected.
            camera_coordinates (np.ndarray): The array of camera coordinates.
            camera_angles (np.ndarray): The array of camera angles.
            vectorized (bool, optional):\
                Whether to use vectorized projection. Defaults to True.
        """
        if not vectorized:
            self.projector: Union[
                PointProjectorCore, PointProjectorCoreVectorized
            ] = PointProjectorCore(image_width, image_height, vertical_offset)
            self.points = np.asarray(points)
            self.camera_coordinates = camera_coordinates
            self.camera_angles = camera_angles
            self.valid_projection_mask = np.ones(
                (len(self.points), len(self.camera_coordinates)), dtype=bool
            )
            self.projected_points = self.project_all(
                self.points, self.camera_coordinates, self.camera_angles
            ).astype(np.int16)
        else:
            self.projector = PointProjectorCoreVectorized(
                image_width, image_height, vertical_offset
            )
            self.projected_points = self.projector.project_on_image(
                np.asarray(points), camera_coordinates, camera_angles
            ).astype(np.int16)

    def project_all(
        self,
        points: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Projects all points onto the images.

        Args:
            points (np.ndarray): The array of 3D points to be projected.
            camera_coordinates (np.ndarray): The array of camera coordinates.
            camera_angles (np.ndarray): The array of camera angles.

        Returns:
            np.ndarray: The array of projected points.
        """
        projected_points = np.zeros((len(points), len(camera_coordinates), 2))
        for i, point in tqdm(
            enumerate(self.points),
            total=len(self.points),
            desc="Projecting points on images",
        ):
            for j, (coordinate, angle) in enumerate(
                zip(camera_coordinates, camera_angles)
            ):
                projection = self.projector.project_on_image(point, coordinate, angle)
                if projection is None:
                    self.valid_projection_mask[i, j] = False
                else:
                    projected_points[i, j] = projection
        return projected_points

    def project(self, point_index: int, image_index: int) -> Optional[np.ndarray]:
        """
        Projects a specific point onto a specific image.

        Args:
            point_index (int): The index of the point to be projected.
            image_index (int): The index of the image.

        Returns:
            Optional[np.ndarray]: The projected point,\
                or None if projection is invalid.
        """
        projection = self.projected_points[point_index, image_index]
        if projection[0] != -1:
            return projection
        return None
