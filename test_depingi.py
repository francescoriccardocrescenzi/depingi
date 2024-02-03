"""Unit tests for depingi.py."""
import unittest as ut
import depingi as dp
import numpy as np
import PIL.Image


def open_image(file_path):
    """Open image from file using PIL."""
    return np.asarray(PIL.Image.open(file_path))


def rgb_pil_image(raw_data):
    """Create PIL image from raw data."""
    return PIL.Image.fromarray(raw_data)

def l_pil_image(raw_data):
    """Create PIL image from raw data."""
    return PIL.Image.fromarray(raw_data, mode="L")


class TestImage(ut.TestCase):
    """Test Image class."""

    def test_getter_and_setter_for_raw(self):
        """Test getter and setter for self.raw."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        im = dp.Image(raw)
        uint8raw = im.raw_as_uint8
        pil_im = rgb_pil_image(uint8raw)
        # pil_im.show()


class TestRGBImage(ut.TestCase):
    """Test LImage class."""

    def test_desaturation_methods(self):
        """Open a colored image and desaturate it in three different ways."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        rgb_im = dp.RGBImage(raw)
        l_im1 = rgb_im.weighted_average_desaturated()
        l_im2 = rgb_im.luminosity_desaturated()
        l_im3 = rgb_im.lightness_desaturated()
        pil_im1 = l_pil_image(l_im1.raw_as_uint8)
        pil_im2 = l_pil_image(l_im2.raw_as_uint8)
        pil_im3 = l_pil_image(l_im3.raw_as_uint8)
        pil_im4 = rgb_pil_image(rgb_im.raw_as_uint8)
        # pil_im1.show()
        # pil_im2.show()
        # pil_im3.show()
        # pil_im4.show()


if __name__ == "__main__":
    ut.main()
