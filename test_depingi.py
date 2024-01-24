"""Unit tests for depingi.py."""
import unittest
import numpy as np
from depingi import LuminanceImage, RGBImage
import PIL.Image


class TestImage(unittest.TestCase):

    def test_RGB_constructor1(self):
        im = RGBImage.from_file("data/plants.jpg")
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_RGB_constructor2(self):
        raw_data = np.asarray(PIL.Image.open("data/plants.jpg"))
        im = RGBImage(raw_data)
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_RGB_to_L_conversion(self):
        RGBim = RGBImage.from_file("data/plants.jpg")
        Lim = RGBim.as_LuminanceImage()
        self.assertTrue(isinstance(Lim, LuminanceImage))
        self.assertEqual(len(Lim.raw.shape), 2)


if __name__ == "__main__":
    unittest.main()
