import os
import unittest

import numpy as np

from src.image_loader.lazy_image_loader import LazyImageLoader


class TestLazyImageLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join("tests", "data", "image_dir")
        self.lazy_image_loader = LazyImageLoader(self.test_dir)

    def test_init(self):
        self.assertEqual(self.lazy_image_loader.image_dir, self.test_dir)
        self.assertIsInstance(self.lazy_image_loader.image_names, list)
        self.assertTrue(
            all(isinstance(i, str) for i in self.lazy_image_loader.image_names)
        )

    def test_get_image(self):
        test_index = 0
        image = self.lazy_image_loader.get_image(test_index)
        self.assertIsInstance(image, np.ndarray)

    def test_len(self):
        self.assertEqual(
            len(self.lazy_image_loader), len(self.lazy_image_loader.image_names)
        )

    def test_no_images_found(self):
        with self.assertRaises(FileNotFoundError):
            LazyImageLoader("/non/existent/directory")


if __name__ == "__main__":
    unittest.main()
