"""A handy Python module for image processing."""
import numpy as np
from enum import Enum, auto
import PIL.Image
from abc import abstractmethod, ABC


class Image(ABC):
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

    @abstractmethod
    def as_PILImage(self):
        """Return the image as a PIL Image."""
        pass


class LuminanceImage(Image):
    """Subclass of Image that handles luminance images."""

    def as_PILImage(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw, mode="L")


class RGBImage(Image):
    """Subclass of Image that handles RGB images."""

    def as_PILImage(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw, mode="RGB")

    def as_LuminanceImage(self, weights: np.ndarray = np.array([1/3, 1/3, 1/3])) -> LuminanceImage:
        """Convert to a luminance image of class LuminanceImage.

        The luminance of each pixel is determined by taking the weighted average of the red, green, and blue components
        according to the provided weights.

        Arguments:
            - weight: array in the form [[wr, wg, wb]], where wr, wg, wb are the weights used for the average.
        """
        # We use a tensor product to take the weighted average.
        float_raw_luminance_image = np.tensordot(self.raw, weights, axes=(2, 0))
        uint8_raw_luminance_image = float_raw_luminance_image.astype(np.uint8)
        return LuminanceImage(uint8_raw_luminance_image)
