"""Unit tests for depingi.py."""
import unittest
import numpy as np
from depingi import Image
import PIL.Image


class TestImage(unittest.TestCase):

    def test_constructor1(self):
        im = Image.from_file("data/plants.jpg")
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_constructor2(self):
        raw_data = np.asarray(PIL.Image.open("data/plants.jpg"))
        im = Image(raw_data)
        self.assertTrue(isinstance(im.raw, np.ndarray))

if __name__ == "__main__":
    unittest.main()
