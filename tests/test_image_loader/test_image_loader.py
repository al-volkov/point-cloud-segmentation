import os
import unittest

import numpy as np

from src.image_loader.image_loader import ImageLoader


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join("tests", "data", "image_dir")
        self.image_loader = ImageLoader(self.test_dir)

    def test_init(self):
        self.assertEqual(self.image_loader.image_dir, self.test_dir)
        self.assertIsInstance(self.image_loader.images, list)
        self.assertTrue(
            all(isinstance(i, np.ndarray) for i in self.image_loader.images)
        )

    def test_get_image(self):
        test_index = 0
        image = self.image_loader.get_image(test_index)
        self.assertIsInstance(image, np.ndarray)

    def test_len(self):
        self.assertEqual(len(self.image_loader), len(self.image_loader.images))

    def test_no_images_found(self):
        with self.assertRaises(FileNotFoundError):
            ImageLoader("/non/existent/directory")


if __name__ == "__main__":
    unittest.main()
