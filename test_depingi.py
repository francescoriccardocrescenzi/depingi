"""Unit tests for depingi.py."""
import unittest
import numpy
import numpy as np
from depingi import Image


class TestImage(unittest.TestCase):

    def test_constructor(self):
        im = Image(filepath="data/plants.jpg")
        self.assertTrue(isinstance(im.raw, np.ndarray))


if __name__ == "__main__":
    unittest.main()
