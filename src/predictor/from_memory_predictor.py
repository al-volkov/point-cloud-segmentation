from functools import lru_cache
from typing import Union

import numpy as np

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader


class FromMemoryPredictor:
    def __init__(
        self, image_loader: Union[ImageLoader, LazyImageLoader], predictions_path: str
    ) -> None:
        self.image_loader = image_loader
        self.predictions = np.load(predictions_path).transpose((0, 2, 1))

    def get_classes(self):
        return [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

    @lru_cache(maxsize=100)
    def predict(self, index: int) -> np.ndarray:
        return self.predictions[index]

    def predict_mulitple(self, indices: np.ndarray) -> np.ndarray:
        return self.predictions[indices]
