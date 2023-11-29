from typing import Union

import numpy as np

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader


class LazyPredictor:
    def __init__(
        self, image_loader: Union[ImageLoader, LazyImageLoader], sem_seg_inferencer
    ) -> None:
        self.image_loader = image_loader
        self.sem_seg_inferencer = sem_seg_inferencer

    def get_classes(self):
        return self.sem_seg_inferencer.classes

    def predict(self, index: int) -> np.ndarray:
        image = self.image_loader.get_image(index)
        return self.sem_seg_inferencer.predict(image)
