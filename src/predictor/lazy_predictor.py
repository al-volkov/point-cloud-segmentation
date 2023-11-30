from typing import Union

import numpy as np

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader


class LazyPredictor:
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
        self, image_loader: Union[ImageLoader, LazyImageLoader], sem_seg_inferencer
    ) -> None:
        """
        Initializes a LazyPredictor object.

        Args:
            image_loader (Union[ImageLoader, LazyImageLoader]):\
                An image loader object used to load images.
            sem_seg_inferencer: The semantic segmentation inferencer\
                object used for prediction.
        """
        self.image_loader = image_loader
        self.sem_seg_inferencer = sem_seg_inferencer

    def get_classes(self):
        """
        Returns the classes used by the semantic segmentation inferencer.

        Returns:
            The classes used by the semantic segmentation inferencer.
        """
        return self.sem_seg_inferencer.classes

    def predict(self, index: int) -> np.ndarray:
        """
        Predicts the semantic segmentation for the image at the given index.

        Args:
            index (int): The index of the image to predict.

        Returns:
            np.ndarray: The predicted semantic segmentation for the image.
        """
        image = self.image_loader.get_image(index)
        return self.sem_seg_inferencer.predict(image)
