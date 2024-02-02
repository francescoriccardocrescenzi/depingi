"""Unit tests for depingi.py."""
import unittest as ut
import depingi as dp
import numpy as np
import PIL.Image


def open_image(file_path):
    """Open image from file using PIL."""
    return np.asarray(PIL.Image.open(file_path))


def pil_image(raw_data):
    """Create PIL image from raw data."""
    return PIL.Image.fromarray(raw_data)


class TestImage(ut.TestCase):
    """Test Image class."""

    def test_getter_and_setter_for_raw(self):
        """Test getter and setter for self.raw."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        im = dp.Image(raw)
        uint8raw = im.raw_as_uint8
        pil_im = pil_image(uint8raw)
        # pil_im.show()


if __name__ == "__main__":
    ut.main()
