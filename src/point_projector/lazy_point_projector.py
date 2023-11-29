from typing import Optional

import numpy as np

from src.point_projector.point_projector_core import PointProjectorCore


class LazyPointProjector:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        points: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> None:
        self.projector = PointProjectorCore(image_width, image_height, vertical_offset)
        self.points = points
        self.camera_coordinates = camera_coordinates
        self.camera_angles = camera_angles

    def project(self, point_index: int, image_index: int) -> Optional[np.ndarray]:
        return self.projector.project_on_image(
            self.points[point_index],
            self.camera_coordinates[image_index],
            self.camera_angles[image_index],
        )
