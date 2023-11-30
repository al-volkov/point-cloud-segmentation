import numpy as np
from mmseg.apis import MMSegInferencer


class MMSegmentor:
    """
    A class used to perform image segmentation using the MMSegInferencer.

    ...

    Attributes
    ----------
    _inferencer : MMSegInferencer
        The MMSegInferencer object used to perform image segmentation.

    Methods
    -------
    predict(image: np.ndarray) -> np.ndarray:
        Performs image segmentation on the provided image and returns the \
            segmentation map.
    """

    def __init__(self, model_name: str) -> None:
        """
        Constructs all the necessary attributes for the MMSegmentor object.

        Parameters
        ----------
            model_name : str
                The name of the model to be used for image segmentation.
        """
        self._inferencer = MMSegInferencer(model=model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
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
        return self._inferencer(image)["predictions"].astype(np.int8)
