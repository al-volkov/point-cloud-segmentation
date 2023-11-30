import numpy as np


def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate various metrics for evaluating the performance of a segmentation model.

    Args:
        true_labels (numpy.ndarray): Array of true labels.
        predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
            - mean_accuracy (float): Mean accuracy.
            - mean_iou (float): Mean intersection over union.
            - accuracies (numpy.ndarray): Array of accuracies for each class.
            - ious (numpy.ndarray):\
                Array of intersection over union values for each class.
    """
    mean_accuracy, accuracies = calculate_accuracy(true_labels, predicted_labels)
    mean_iou, ious = calculate_iou(true_labels, predicted_labels)
    return {
        "mean_accuracy": mean_accuracy,
        "mean_iou": mean_iou,
        "accuracies": accuracies,
        "ious": ious,
    }


def calculate_iou(true_labels, predicted_labels):
    """
    Calculates the Intersection over Union (IoU) metric for semantic segmentation.

    Args:
        true_labels (numpy.ndarray): Array of true labels.
        predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
        float: Mean IoU value.
        numpy.ndarray: Array of IoU values for each label.
    """
    max_label = 18  # labels from -1 to 18
    confusion_matrix = np.zeros((max_label + 2, max_label + 2), dtype=np.int64)
    ious = np.zeros(max_label + 2)
    ious -= 1

    for i in range(-1, max_label + 1):
        for j in range(-1, max_label + 1):
            confusion_matrix[i + 1, j + 1] = np.logical_and(
                true_labels == i, predicted_labels == j
            ).sum()
    for i in range(-1, max_label + 1):
        intersection = confusion_matrix[i + 1, i + 1]
        union = (
            confusion_matrix[i + 1, :].sum()
            + confusion_matrix[:, i + 1].sum()
            - intersection
        )
        if union == 0:
            continue
        iou = intersection / union
        ious[i + 1] = iou

    valid_ious = ious[(ious > 0) & (np.arange(-1, max_label + 1) != -1)]
    mean_iou = np.mean(valid_ious)
    return mean_iou, ious


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate the mean accuracy and individual accuracies\
        for a given set of true labels and predicted labels.

    Parameters:
    true_labels (numpy.ndarray): Array of true labels.
    predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
    mean_accuracy (float): Mean accuracy calculated from the individual accuracies.
    accuracies (numpy.ndarray): Array of individual accuracies for each label.
    """
    max_label = 18
    confusion_matrix = np.zeros((max_label + 2, max_label + 2), dtype=np.int64)
    accuracies = np.zeros(max_label + 2)
    accuracies -= 1
    for i in range(-1, max_label + 1):
        for j in range(-1, max_label + 1):
            confusion_matrix[i + 1, j + 1] = np.logical_and(
                true_labels == i, predicted_labels == j
            ).sum()
    for i in range(-1, max_label + 1):
        true_positive = confusion_matrix[i + 1, i + 1]
        false_positive = confusion_matrix[:, i + 1].sum() - true_positive
        false_negative = confusion_matrix[i + 1, :].sum() - true_positive
        total = true_positive + false_positive + false_negative
        if total == 0:
            continue
        accuracy = true_positive / total
        accuracies[i + 1] = accuracy

    valid_accuracies = accuracies[accuracies > 0]
    mean_accuracy = np.mean(valid_accuracies)

    return mean_accuracy, accuracies
