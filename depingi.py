"""A handy Python module for image processing."""
import numpy as np
from enum import Enum, auto
import PIL.Image


class InitMethod(Enum):
    """Enumeration used to choose initialization method for instances of Image."""
    FILE = auto()
    RAW = auto()


class Image:
    """This class stores images as numpy arrays."""

    def __init__(self, filepath: str = None, raw_image: np.ndarray = None, method: InitMethod = InitMethod.FILE):
        """Initialize a new instance of Image.

        There are two initialization methods:
            - by file: set method = InitMethod.FILE and provide the file path using hte keyword argument filepath;
            - by raw data: set method = InitMethod.RAW and provide the row image as a numpy array eiter in RGB form or
            in greyscale.

        Arguments:
            - filepath: path to the file containing the image in any format compatible with the PIL library
            - raw_image: numpy array containing the image in RGB or greyscale format
            - method: initialization method provided using the InitMethod Enum
        """
        if method == InitMethod.FILE:
            self.raw = np.asarray(PIL.Image.open(filepath))
        elif method == InitMethod.RAW:
            self.raw = raw_image

    def as_PIL_Image(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw)