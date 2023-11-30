from typing import Optional

import numpy as np

from src.point_projector.point_projector_core import PointProjectorCore


class LazyPointProjector:
    """
    Performs image segmentation on the provided image.

    Parameters
    ----------
        image : np.ndarray
            The image to be segmented.

    Returns
    -------
        np.ndarray
            The segmentation map of the image.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        points: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> None:
        """
        Initializes a LazyPointProjector object.

        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image.
            vertical_offset (int): The vertical offset of the image.
            points (np.ndarray): The array of points to be projected.
            camera_coordinates (np.ndarray): The array of camera coordinates.
            camera_angles (np.ndarray): The array of camera angles.
        """
        self.projector = PointProjectorCore(image_width, image_height, vertical_offset)
        self.points = points
        self.camera_coordinates = camera_coordinates
        self.camera_angles = camera_angles

    def project(self, point_index: int, image_index: int) -> Optional[np.ndarray]:
        """
        Projects a point onto an image.

        Args:
            point_index (int): The index of the point to be projected.
            image_index (int): The index of the image.

        Returns:
            Optional[np.ndarray]: The projected point as a numpy array,\
                or None if the projection fails.
        """
        return self.projector.project_on_image(
            self.points[point_index],
            self.camera_coordinates[image_index],
            self.camera_angles[image_index],
        )
