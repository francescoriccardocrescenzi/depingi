"""A handy Python module for image processing."""
import numpy as np
from enum import Enum, auto
import PIL.Image


class Image:
    """This class stores images as numpy arrays."""

    def __init__(self, raw_image):
        """Initialize a new instance of Image by providing the raw data as a numpy array."""
        self.raw = raw_image

    @classmethod
    def from_file(cls, file_path: str):
        """Initialize a new instance of Image reading the data from a file.

        The image is decoded using PIL.Image.open and then converted to a numpy array.
        """
        return cls(np.asarray(PIL.Image.open(file_path)))

    def as_PIL_Image(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw)