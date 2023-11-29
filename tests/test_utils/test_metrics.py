import unittest

import numpy as np

from src.utils.metrics import calculate_accuracy


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.true_labels = np.array(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        )
        self.predicted_labels = np.array(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        )

    def test_calculate_accuracy(self):
        mean_accuracy, accuracies = calculate_accuracy(
            self.true_labels, self.predicted_labels
        )
        self.assertEqual(mean_accuracy, 1.0)
        self.assertTrue((accuracies == 1.0).all())


if __name__ == "__main__":
    unittest.main()
