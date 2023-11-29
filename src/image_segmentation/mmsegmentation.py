import numpy as np
from mmseg.apis import MMSegInferencer


class MMSegmentor:
    def __init__(self, model_name):
        self._inferencer = MMSegInferencer(model=model_name)

    def predict(self, image):
        return self._inferencer(image)["predictions"].astype(np.int8)
