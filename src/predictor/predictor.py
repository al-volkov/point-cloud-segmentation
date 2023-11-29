from typing import Union

import numpy as np
from tqdm import tqdm

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader
from src.image_segmentation.mmsegmentation import MMSegmentor
from src.image_segmentation.panoptic_deeplab import PanopticDeepLabSegmentor


class Predictor:
    def __init__(
        self,
        image_loader: Union[ImageLoader, LazyImageLoader],
        sem_seg_inferencer: Union[MMSegmentor, PanopticDeepLabSegmentor],
    ) -> None:
        self.image_loader = image_loader
        self.sem_seg_inferencer = sem_seg_inferencer
        self.predictions_list = []
        for image_index in tqdm(
            range(len(self.image_loader)),
            total=len(self.image_loader),
            desc="Predicting labels for images",
        ):
            image = self.image_loader.get_image(image_index)
            self.predictions_list.append(self.sem_seg_inferencer.predict(image))
        self.predictions = np.array(self.predictions_list).astype(np.int8)

    def get_classes(self):
        return self.sem_seg_inferencer.classes

    def predict(self, index: int) -> np.ndarray:
        return self.predictions[index]
