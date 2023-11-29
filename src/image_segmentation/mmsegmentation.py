import numpy as np
from mmseg.apis import MMSegInferencer


class MMSegmentor:
    def __init__(self, model_name: str) -> None:
        self._inferencer = MMSegInferencer(model=model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self._inferencer(image)["predictions"].astype(np.int8)
