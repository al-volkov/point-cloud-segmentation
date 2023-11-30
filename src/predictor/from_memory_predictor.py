from functools import lru_cache
from typing import Union

import numpy as np

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader


class FromMemoryPredictor:
    """
    A class that represents a predictor for semantic segmentation from memory.

    Args:
        image_loader (Union[ImageLoader, LazyImageLoader]):\
            An image loader object used to load images.
        predictions_path (str): The path to the predictions file.

    Attributes:
        image_loader (Union[ImageLoader, LazyImageLoader]):\
            An image loader object used to load images.
        predictions (np.ndarray): The predictions loaded from the predictions file.

    Methods:
        get_classes(): Returns a list of classes.
        predict(index: int) -> np.ndarray: Returns the prediction for a given index.
        predict_multiple(indices: np.ndarray) -> np.ndarray:\
            Returns the predictions for multiple indices.
    """

    def __init__(
        self, image_loader: Union[ImageLoader, LazyImageLoader], predictions_path: str
    ) -> None:
        """
        Initialize the FromMemoryPredictor object.

        Args:
            image_loader (Union[ImageLoader, LazyImageLoader]):\
                An instance of ImageLoader or LazyImageLoader.
            predictions_path (str): The path to the predictions file.

        Raises:
            FileNotFoundError: If the predictions file is not found.
        """
        self.image_loader = image_loader
        try:
            self.predictions = np.load(predictions_path).transpose((0, 2, 1))
        except FileNotFoundError:
            raise FileNotFoundError(f"Predictions file {predictions_path} not found.")

    def get_classes(self):
        """
        Returns a list of classes for point cloud segmentation.

        Returns:
            list: A list of classes including "road", "sidewalk", "building",\
                "wall", "fence", "pole",
            "traffic light", "traffic sign", "vegetation", "terrain", "sky",\
                "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", and "bicycle".
        """
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
        """
        Predicts the output for a given index.

        Args:
            index (int): The index of the prediction to retrieve.

        Returns:
            np.ndarray: The predicted output for the given index.
        """
        return self.predictions[index]

    def predict_multiple(self, indices: np.ndarray) -> np.ndarray:
        """
        Predicts multiple values based on the given indices.

        Args:
            indices (np.ndarray): The indices of the values to predict.

        Returns:
            np.ndarray: The predicted values corresponding to the given indices.
        """
        return self.predictions[indices]
