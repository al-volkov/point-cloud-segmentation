import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects import panoptic_deeplab


class PanopticDeepLabSegmentor:
    """
    A class used to perform image segmentation using the Panoptic-DeepLab\
        model from Detectron2.

    ...

    Attributes
    ----------
    cfg : detectron2.config.CfgNode
        The configuration for the Panoptic-DeepLab model.
    predictor : detectron2.engine.DefaultPredictor
        The predictor object used to perform image segmentation.
    metadata : detectron2.data.MetadataCatalog
        The metadata associated with the training dataset.
    classes : list
        The list of class names in the training dataset.

    Methods
    -------
    predict(image: np.ndarray) -> np.ndarray:
        Performs image segmentation on the provided image and returns\
            the segmentation map.
    """

    def __init__(self, cfg_path: str, weights_path: str) -> None:
        """
        Constructs all the necessary attributes for the PanopticDeepLabSegmentor object.

        Parameters
        ----------
            cfg_path : str
                The path to the configuration file for the Panoptic-DeepLab model.
            weights_path : str
                The path to the weights file for the Panoptic-DeepLab model.
        """
        self.cfg = get_cfg()
        panoptic_deeplab.add_panoptic_deeplab_config(self.cfg)
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.classes = self.metadata.stuff_classes

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
        sem_seg_results = self.predictor(image)["sem_seg"]
        result_classes = sem_seg_results.argmax(dim=0).to("cpu")
        return result_classes.numpy().astype(np.int8)
