import argparse
import os
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from src.image_loader.image_loader import ImageLoader
from src.image_loader.lazy_image_loader import LazyImageLoader
from src.point_cloud_segmentor import PointCloudSegmentor
from src.predictor.from_memory_predictor import FromMemoryPredictor
from src.predictor.lazy_predictor import LazyPredictor
from src.predictor.predictor import Predictor
from src.utils.read_config import read_config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--display", action="store_true", default=False)
    args = parser.parse_args()
    config = read_config(args.config_path)

    try:
        point_cloud = o3d.io.read_point_cloud(config["point_cloud_path"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Point cloud file not found at '{config['point_cloud_path']}'"
        )

    try:
        true_labels = (
            o3d.t.io.read_point_cloud(config["point_cloud_path"])
            .point["scalar_Classification"]
            .numpy()
        )
        true_labels = np.squeeze(true_labels)
        true_labels -= 1
    except KeyError:
        raise FileNotFoundError(
            f"Point cloud file at '{config['point_cloud_path']}' does not contain labels"
        )

    if config["images"]["read_all"]:
        image_loader: Union[ImageLoader, LazyImageLoader] = ImageLoader(
            config["images"]["image_dir"]
        )
    else:
        image_loader = LazyImageLoader(config["images"]["image_dir"])

    try:
        reference = pd.read_csv(config["reference_path"], sep="\t")
        reference = reference[
            reference["file_name"].isin(
                os.path.splitext(os.path.basename(image_path))[0]
                for image_path in image_loader.image_names
            )
        ]
        camera_coordinates = reference.loc[:, "projectedX[m]":"projectedZ[m]"].values  # type: ignore
        camera_angles = reference.loc[:, "roll[deg]":"heading[deg]"].values  # type: ignore
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Reference file not found at '{config['reference_path']}'"
        )

    if config["predictor"]["type"] == "memory":
        predictor: Union[
            Predictor, FromMemoryPredictor, LazyPredictor
        ] = FromMemoryPredictor(image_loader, config["predictor"]["predictions_path"])
    elif config["predictor"]["type"] == "precompute":
        sem_seg_inferencer = None
        predictor = Predictor(image_loader, sem_seg_inferencer)  # type: ignore
    elif config["predictor"]["type"] == "lazy":
        sem_seg_inferencer = None
        predictor = LazyPredictor(image_loader, sem_seg_inferencer)
    else:
        raise ValueError("Error: Invalid predictor type specified in config file")

    tree = cKDTree(camera_coordinates)

    closest_images = tree.query(
        point_cloud.points, k=config["algorithm"]["compute_closest"]
    )[1].astype(np.int16)

    pcs = PointCloudSegmentor(
        point_cloud=point_cloud,
        camera_coordinates=camera_coordinates,
        camera_angles=camera_angles,
        image_loader=image_loader,
        image_width=config["images"]["image_width"],
        image_height=config["images"]["image_height"],
        vertical_offset=config["images"]["vertical_offset"],
        predictor=predictor,
        closest_images=closest_images,
        true_labels=true_labels,
    )

    predicted_labels = pcs.predict(
        batch_size=config["algorithm"]["batch_size"],
        voting_images=config["algorithm"]["voting_images"],
        drop_first=config["algorithm"]["drop_first"],
    )

    pcs.evaluate(predicted_labels)

    if args.display:
        predicted_labels += 3
        num_labels = np.unique(predicted_labels).shape[0]
        colors = plt.set_cmap("tab10")(predicted_labels / num_labels)[:, :3]
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])
