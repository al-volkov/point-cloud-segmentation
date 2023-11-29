from itertools import chain

import numpy as np
import open3d as o3d
from scipy.stats import mode
from tabulate import tabulate
from tqdm import tqdm

from src.point_projector.point_projector import PointProjector
from src.utils.metrics import calculate_metrics


class PointCloudSegmentor:
    def __init__(
        self,
        point_cloud: o3d.geometry.PointCloud,
        camera_coordinates: np.ndarray,
        camera_angles: np.ndarray,
        image_loader,
        image_width: int,
        image_height: int,
        vertical_offset: int,
        predictor,
        closest_images,
        true_labels=None,
    ) -> None:
        self.point_cloud = point_cloud
        self.image_loader = image_loader
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_offset = vertical_offset
        self.camera_coordinates = camera_coordinates
        self.camera_angles = camera_angles
        self.predictor = predictor
        self.closest_images = closest_images
        self.true_labels = true_labels

    def predict(self, batch_size, voting_images, drop_first):
        total_points = len(self.point_cloud.points)
        segmented = []
        for start_index in tqdm(
            range(0, total_points, batch_size), desc="Iterating over batches"
        ):
            end_idx = min(start_index + batch_size, total_points)
            segmented_batch = self.predict_batch(
                start_index, end_idx, voting_images, drop_first
            )
            segmented.append(segmented_batch)

        result_labels = list(chain.from_iterable(segmented))
        result_labels = np.array(result_labels)

        return result_labels.astype(np.int8)

    def predict_batch(self, start_idx, end_idx, voting_images, drop_first):
        labels = []
        projector = PointProjector(
            self.image_width,
            self.image_height,
            self.vertical_offset,
            self.point_cloud.points[start_idx:end_idx],
            self.camera_coordinates,
            self.camera_angles,
        )
        for point_index in range(start_idx, end_idx):
            if len(self.true_labels) != 0:
                if self.true_labels[point_index] == -1:
                    labels.append(-1)
                    continue
            current_closest_images = self.closest_images[point_index][drop_first:]
            current_labels = []
            current_predictions = [
                self.predictor.predict(image_index)
                for image_index in current_closest_images
            ]  # this is faster that vectorized for some reason
            for i, image_index in enumerate(current_closest_images):
                prediction = current_predictions[i]
                projected_point = projector.project(
                    point_index - start_idx, image_index
                )
                if projected_point is not None:
                    label = prediction[projected_point[0], projected_point[1]]
                    if label != 10:
                        current_labels.append(label)
                if len(current_labels) >= voting_images:
                    break
            if current_labels:
                labels.append(mode(current_labels).mode)
            else:
                labels.append(-1)
        return labels

    def evaluate(self, predicted_labels):
        results = calculate_metrics(self.true_labels, predicted_labels)
        table_data = [
            (label, results["accuracies"][label + 1], results["ious"][label + 1])
            for label in range(-1, len(results["ious"]) - 1)
        ]

        table_data = filter(lambda x: x[0] != -1 and x[1] > 0 and x[2] > 0, table_data)

        headers = ["Label", "Accuracy", "IoU"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        print(table)

        table_data = [
            ["Metric", "Value"],
            ["Mean Accuracy", results["mean_accuracy"]],
            ["Mean IOU", results["mean_iou"]],
        ]

        table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
        print(table)
