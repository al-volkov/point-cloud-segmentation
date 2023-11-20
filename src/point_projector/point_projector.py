import numpy as np
from tqdm import tqdm

from src.point_projector.point_projector_core import PointProjectorCore
from src.point_projector.point_projector_core_vectorized import (
    PointProjectorCoreVectorized,
)


class PointProjector:
    __slots__ = (
        "projector",
        "points",
        "camera_coordinates",
        "camera_angles",
        "projected_points",
        "vectorized",
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
        self.vectorized = vectorized
        if not vectorized:
            self.projector = PointProjectorCore(
                image_width, image_height, vertical_offset
            )
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
            )

    def project_all(
        self,
        points: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
    ) -> np.ndarray:
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

    def project(self, point_index: int, image_index: int) -> np.ndarray:
        projection = self.projected_points[point_index, image_index]
        if projection[0] != -1:
            return projection
        return None
