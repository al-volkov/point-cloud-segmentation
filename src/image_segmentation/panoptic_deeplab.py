import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects import panoptic_deeplab


class PanopticDeepLabSegmentor:
    def __init__(self, cfg_path, weights):
        self.cfg = get_cfg()
        panoptic_deeplab.add_panoptic_deeplab_config(self.cfg)
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = weights
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.classes = self.metadata.stuff_classes

    def predict(self, image):
        sem_seg_results = self.predictor(image)["sem_seg"]
        result_classes = sem_seg_results.argmax(dim=0).to("cpu")
        return result_classes.numpy().astype(np.int8)
