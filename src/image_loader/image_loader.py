import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


class ImageLoader:
    """
    A class used to load and manage images from a specified directory.

    ...

    Attributes
    ----------
    image_dir : str
        The directory path where the images are located.
    images : list
        A list of loaded images.
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
        Initialize the ImageLoader class.

        Args:
            image_dir (str): The directory path where the images are located.

        Raises:
            FileNotFoundError: If no images are found in the specified directory.
        """
        self.image_dir = image_dir
        self.images = []
        self.image_names = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        if len(self.image_names) == 0:
            raise FileNotFoundError(f"No images found in directory '{self.image_dir}'")
        for image_path in tqdm(
            self.image_names, total=len(self.image_names), desc="Loading images"
        ):
            self.images.append(cv2.imread(image_path))

    def get_image(self, index: int) -> np.ndarray:
        """
        Get the image at the specified index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        return self.images[index]

    def __len__(self) -> int:
        """
        Get the number of images in the ImageLoader.

        Returns:
            int: The number of images.
        """
        return len(self.images)
