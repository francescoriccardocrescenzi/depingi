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
        raw_data = np.asarray(PIL.Image.open(file_path))
        return cls(raw_data)

    @abstractmethod
    def as_PILImage(self):
        """Return the image as a PIL Image."""
        pass

    def histogram_bin_number(self) -> int:
        """Helper method which provides the number of bins of the histogram of the image.

        The default bin number is set to sqrt(npix), where npix is the number of pixels of the image.
        """
        npix = self.raw.shape[0]*self.raw.shape[1]
        return int(np.sqrt(npix))

    @abstractmethod
    def histogram(self, bin_number: int = None) -> np.ndarray:
        """Return the histogram of the image."""
        pass


class LuminanceImage(Image):
    """Subclass of Image that handles luminance images."""

    def as_PILImage(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw, mode="L")

    def histogram(self, bin_number: int = None):
        """Provide the intensity histogram of the picture.

        The output of numpy.histogram is returned as it.
        """
        if bin_number is None:
            bin_number = self.histogram_bin_number()
        range = (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
        return np.histogram(self.raw, bins=bin_number, range=range)

    # TODO: Take min_val and max_val from a statistical distribution to reduce the likelihood of the method faliing.
    # TODO: Account for possible zero division when min_val and max_val are equal.
    def apply_contrast_stretching(self):
        """
        Increase the contrast of the image by applying contrast stretching.

        Apply to each pixel the following mathematical formula:
        P_new = ((P_old - min_val)/(max_val - min_val))*255.
        """
        max_val = self.raw.max()
        min_val = self.raw.min()
        stretched_raw = ((self.raw-min_val)/(max_val-min_val)*np.iinfo(np.uint8).max).astype(np.uint8)
        self.raw = stretched_raw


class RGBImage(Image):
    """Subclass of Image that handles RGB images."""

    def red_component(self):
        """Return red component of self.raw."""
        return self.raw[:, :, 0]

    def green_component(self):
        """Return green component of self.raw."""
        return self.raw[:, :, 1]

    def blue_component(self):
        """Return blue component of self.raw."""
        return self.raw[:, :, 2]

    def as_PILImage(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw, mode="RGB")

    def weighted_average_desaturated(self, weights: np.ndarray = np.array([1/3, 1/3, 1/3])) -> LuminanceImage:
        """Convert to a luminance image using a weighted average of the RGB channels.

        The luminance of each pixel is determined by taking the weighted average of the red, green, and blue components
        according to the provided weights.

        Arguments:
            - weight: array in the form [wr, wg, wb], where wr, wg, wb are the weights used for the average.
        """
        # We use a tensor product to take the weighted average.
        float_raw_luminance_image = np.tensordot(self.raw, weights, axes=(2, 0))
        uint8_raw_luminance_image = float_raw_luminance_image.astype(np.uint8)
        return LuminanceImage(uint8_raw_luminance_image)

    def luminosity_desaturated(self) -> LuminanceImage:
        """Convert to a luminance image using the luminosity method.

        Apply Image.weighted_average_desaturated with weights [0.21, 0.72, 0.07].
        """
        return self.weighted_average_desaturated(np.array([0.21, 0.72, 0.07]))

    def lightness_desaturated(self) -> LuminanceImage:
        """Convert to a luminance image using the lightness method.

        The luminance of each pixel is determined as (min(R,G,B) + max(R,G,B))/2.
        """
        float_raw_luminance_image = (np.amax(self.raw, axis=2) + np.amin(self.raw, axis=2))//2
        uint8_raw_luminance_image = float_raw_luminance_image.astype(np.uint8)
        return LuminanceImage(uint8_raw_luminance_image)

    # TODO: Test this method.
    # TODO: Write methods to get the histograms associated with the RGB channels separately.
    def histogram(self, bin_number: int = None):
        """Provide the 3-dimensional color histogram of the picture.

        The output of numpy.histogramdd is returned as is.
        """
        if bin_number is None:
            bin_number = self.histogram_bin_number()
        range = (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
        # Create a new array each row of which represents one pixel in RGB coordinates. This is necessary because
        # np.histogramdd only accepts data in this form.
        pixels = self.raw.reshape(-1, self.raw.shape[-1])
        return np.histogramdd(pixels, bins=bin_number, range=range)

