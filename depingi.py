"""A handy Python module for image processing."""
import numpy as np


class Image:
    """Class used to open and process images.

    The image is stored in raw form as numpy ndarray.
    """

    # CONSTANTS

    # Alias the uint8 maximum and minimum values for increased legibility.
    UINT8MAX = np.iinfo(np.uint8).max
    UINT8MIN = np.iinfo(np.uint8).min

    # INITIALIZATION

    def __init__(self, raw_data: np.ndarray):
        """Create a new instance of Image from raw image data."""
        self.raw = raw_data

    # GETTERS AND SETTERS

    @property
    def raw(self) -> np.ndarray:
        """Getter for self.raw."""
        return self._raw

    @raw.setter
    def raw(self, raw_data: np.ndarray):
        """Setter for self.raw."""
        raw_data = raw_data.astype(np.float32)/self.UINT8MAX
        self._raw = raw_data

    @property
    def raw_as_uint8(self):
        return (self.raw*self.UINT8MAX).astype(np.uint8)


class LImage(Image):
    """Subclass of Image to open and process luminance images.

    The images are stored as 2d numpy arrays and the luminance of each pixel is represented as a float 0 < lum < 1.
    """
    pass


class RGBImage(Image):
    """Subclass of Image to open and process RGB images.

    The images are stored as 3d numpy arrays.
    """
    pass


