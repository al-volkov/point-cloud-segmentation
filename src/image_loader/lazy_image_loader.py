import glob
import os
from typing import List

import cv2
import numpy as np


class LazyImageLoader:
    """
    A class used to load images from a specified directory in a lazy manner.

    ...

    Attributes
    ----------
    image_dir : str
        The directory path where the images are located.
    image_names : list
        A list of image file paths sorted in ascending order.

    Methods
    -------
    get_image(index: int) -> np.ndarray:
        Returns the image at the specified index.
    __len__() -> int:
        Returns the number of images loaded.
    """

    def __init__(self, image_dir: str) -> None:
        """
        Initializes a LazyImageLoader object.

        Args:
            image_dir (str): The directory path where the images are located.

        Raises:
            FileNotFoundError: If no images are found in the specified directory.
        """
        self.image_dir = image_dir
        self.image_names: List[str] = sorted(
            glob.glob(os.path.join(self.image_dir, "*.jpg"))
        )
        if len(self.image_names) == 0:
            raise FileNotFoundError(f"No images found in directory '{self.image_dir}'")

    def get_image(self, index: int) -> np.ndarray:
        """
        Retrieves the image at the specified index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        image_path: str = self.image_names[index]
        return cv2.imread(image_path)

    def __len__(self) -> int:
        """
        Returns the number of images in the image loader.

        Returns:
            int: The number of images.
        """
        return len(self.image_names)
