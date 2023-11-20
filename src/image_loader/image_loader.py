import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


class ImageLoader:
    def __init__(self, image_dir: str) -> None:
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
        return self.images[index]

    def __len__(self) -> int:
        return len(self.images)
