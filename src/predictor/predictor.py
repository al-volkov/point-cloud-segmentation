from typing import Union

import numpy as np
from tqdm import tqdm

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader
from src.image_segmentation.mmsegmentation import MMSegmentor
from src.image_segmentation.panoptic_deeplab import PanopticDeepLabSegmentor


class Predictor:
    """
    A class used to perform prediction of semantic segmentation on images.

    ...

    Attributes
    ----------
    image_loader : Union[ImageLoader, LazyImageLoader]
        The image loader object used to load images.
    sem_seg_inferencer : Union[MMSegmentor, PanopticDeepLabSegmentor]
        The semantic segmentation inferencer object used for prediction.
    predictions_list : list
        A list of predictions for each image.
    predictions : np.ndarray
        An array of predictions for each image.

    Methods
    -------
    get_classes() -> list:
        Returns the classes used by the semantic segmentation inferencer.
    predict(index: int) -> np.ndarray:
        Predicts the semantic segmentation for the image at the given index.
    """

    def __init__(
        self,
        image_loader: Union[ImageLoader, LazyImageLoader],
        sem_seg_inferencer: Union[MMSegmentor, PanopticDeepLabSegmentor],
    ) -> None:
        """
        Initializes a Predictor object.

        Args:
            image_loader (Union[ImageLoader, LazyImageLoader]):\
                An image loader object used to load images.
            sem_seg_inferencer (Union[MMSegmentor, PanopticDeepLabSegmentor]):\
                A semantic segmentation inferencer object used for prediction.
        """
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
        """
        Returns the classes used by the semantic segmentation inferencer.

        Returns:
            List[str]: A list of class names.
        """
        return self.sem_seg_inferencer.classes

    def predict(self, index: int) -> np.ndarray:
        """
        Predicts the segmentation labels for the image at the specified index.

        Args:
            index (int): The index of the image to predict.

        Returns:
            np.ndarray: An array of predicted segmentation labels.
        """
        return self.predictions[index]
