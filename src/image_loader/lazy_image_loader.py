import glob
import os
from typing import List

import cv2
import numpy as np


class LazyImageLoader:
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.image_names: List[str] = sorted(
            glob.glob(os.path.join(self.image_dir, "*.jpg"))
        )
        if len(self.image_names) == 0:
            raise FileNotFoundError(f"No images found in directory '{self.image_dir}'")

    def get_image(self, index: int) -> np.ndarray:
        image_path: str = self.image_names[index]
        return cv2.imread(image_path)

    def __len__(self) -> int:
        return len(self.image_names)
