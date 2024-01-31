"""A handy Python module for image processing."""
import numpy as np
import PIL.Image
from abc import ABC, abstractmethod


class ImageHistogram:
    """Helper class used to store and display the histograms for Image instances."""
    pass


class LuminanceImageHistogram(ImageHistogram):
    """Helper class used to store and display the histograms for LuminanceImage instances."""

    def __init__(self, values: np.ndarray, bins: np.ndarray):
        """Initialize a new instance of LuminanceImageHistogram."""
        self.values = values
        self.bins = bins


class RGBImageHistogram(ImageHistogram):
    """Helper class used to store and display the histograms for RGBImage instances."""

    def __init__(self, red_values: np.ndarray, red_bins: np.ndarray, green_values: np.ndarray, green_bins: np.ndarray,
                 blue_values: np.ndarray, blue_bins: np.ndarray):
        """Initialize a new instance of RGBImageHistogram."""
        self.red_values = red_values
        self.red_bins = red_bins
        self.green_values = green_values
        self.green_bins = green_bins
        self.blue_values = blue_values
        self.blue_bins = blue_bins


class Image:
    """This class stores images as numpy arrays. 
    
    This class is not to be instatiated directly. Use the subclasses LuminanceImage and RGBImage instead.
    """

    # Alias the uint8 maximum and minimum values for increased legibility.
    UINT8MAX = np.iinfo(np.uint8).max
    UINT8MIN = np.iinfo(np.uint8).min

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

    def histogram_bin_number(self) -> int:
        """Helper method intended for internal used only.

        This method provides the number of bins of the histogram of the image.
        The default bin number is set to sqrt(npix), where npix is the number of pixels of the image.
        """
        npix = self.raw.shape[0] * self.raw.shape[1]
        return int(np.sqrt(npix))

    @classmethod
    def apply_contrast_stretching_to_raw_luminance_image(cls, raw: np.ndarray,
                                                         lower_percentile_rank: int,
                                                         upper_percentile_rank: int) -> np.ndarray:
        """Helper method intended for internal use only.

        The method applies contrast stretching to a raw luminance image encoded as a Numpy uint8 array.
        The following formula is applied to each pixel:
        P_new = ((P_old - lower_percentile)/(upper_percentile - lower_percentile))*255.

        Arguments:
            -- lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            -- upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both arguments should be integers between 1 and 99.
        """
        lower_percentile = np.percentile(raw, lower_percentile_rank)
        upper_percentile = np.percentile(raw, upper_percentile_rank)
        if upper_percentile > lower_percentile:
            new_raw = ((raw - lower_percentile) / (upper_percentile - lower_percentile) * cls.UINT8MAX)
            # After the transformation is applied, clip new_raw in place to the max and min values of uint8.
            # We use the out parameter to carry out the clipping in place.
            np.clip(new_raw, cls.UINT8MIN, cls.UINT8MAX, out=new_raw)
            # New raw has been created as a float array but needs to be converted to uint8.
            new_raw = new_raw.astype(np.uint8)
        else:
            new_raw = raw
        return new_raw

    @classmethod
    def histogram_from_raw_luminance_image(cls, raw: np.ndarray, bin_number: int) -> (np.ndarray, np.ndarray):
        """Helper method only intended for internal use.

        Returns the histogram of raw with the given bin number and in the given range as outputted by np.histogram.
        """
        return np.histogram(raw, bins=bin_number, range=(cls.UINT8MIN, cls.UINT8MAX))

    def show(self):
        """Convert the image to a PIL Image and display it on the screen."""
        self.as_PILImage().show()

    @abstractmethod
    def as_PilImage(self):
        pass

    @abstractmethod
    def histogram(self, bin_number: int = None) -> ImageHistogram:
        """Returns the histogram of the image.

        Arguments:
            - bin_number: the number of bins that the histogram will have; if set to None,
                self.histogram_bin_number() will be used instead.
        """
        pass


class LuminanceImage(Image):
    """Subclass of Image that handles luminance images."""

    def as_PILImage(self):
        """Return the image as a PIL Image."""
        return PIL.Image.fromarray(self.raw, mode="L")

    def histogram(self, bin_number: int = None) -> LuminanceImageHistogram:
        if bin_number is None:
            bin_number = self.histogram_bin_number()
        values, bins = self.histogram_from_raw_luminance_image(self.raw, bin_number)
        return LuminanceImageHistogram(values, bins)

    def apply_contrast_stretching(self, lower_percentile_rank: int = 1, upper_percentile_rank: int = 99) \
            -> "LuminanceImage":
        """
        Create a new LuminanceImage by applying contrast stretching.

        Apply to each pixel the following mathematical formula:
        P_new = ((P_old - lower_percentile)/(upper_percentile - lower_percentile))*255.
        If lower_percentile == upper_percentile, the formula cannot be applied as the image is a solid shade;
        thus, a copy of the image is created without applying any transformation.

        Arguments:
            -- lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            -- upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both arguments should be integers between 1 and 99.
        """
        return LuminanceImage(Image.apply_contrast_stretching_to_raw_luminance_image(self.raw,
                                                                                     lower_percentile_rank,
                                                                                     upper_percentile_rank))


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

    def weighted_average_desaturated(self, weights: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])) -> LuminanceImage:
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
        float_raw_luminance_image = (np.amax(self.raw, axis=2) + np.amin(self.raw, axis=2)) // 2
        uint8_raw_luminance_image = float_raw_luminance_image.astype(np.uint8)
        return LuminanceImage(uint8_raw_luminance_image)

    # TODO: Test this method.
    # TODO: Write methods to get the histograms associated with the RGB channels separately.
    def histogram(self, bin_number: int = None):
        if bin_number is None:
            bin_number = self.histogram_bin_number()
        red_values, red_bins = self.histogram_from_raw_luminance_image(self.red_component(), bin_number)
        green_values, green_bins = self.histogram_from_raw_luminance_image(self.green_component(), bin_number)
        blue_values, blue_bins = self.histogram_from_raw_luminance_image(self.blue_component(), bin_number)
        return RGBImageHistogram(red_values, red_bins,  green_values, green_bins, blue_values, blue_bins)

    def apply_contrast_stretching(self, lower_percentile_rank: int = 1, upper_percentile_rank: int = 99) \
            -> "RGBImage":
        """
        Create a new RGBImage by applying contrast stretching.

        Apply to each channel of each pixel the following mathematical formula:
        P_new = ((P_old - lower_percentile)/(upper_percentile - lower_percentile))*255.
        The percentile ranks are the same for each channel but the actual percentiles depend on the specific channel.
        If lower_percentile == upper_percentile for some channel, the formula cannot be applied to that specific channel
        and thus such channel is left unchanged.

        Arguments:
            -- lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            -- upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both arguments should be integers between 1 and 99.
        """
        # Apply the transformation to each channel using a helper method and stack the channels together to obtain a
        # well-formatted new raw image.
        new_red_component = self.apply_contrast_stretching_to_raw_luminance_image(self.red_component(),
                                                                                  lower_percentile_rank,
                                                                                  upper_percentile_rank)
        new_green_component = self.apply_contrast_stretching_to_raw_luminance_image(self.green_component(),
                                                                                    lower_percentile_rank,
                                                                                    upper_percentile_rank)
        new_blue_component = self.apply_contrast_stretching_to_raw_luminance_image(self.blue_component(),
                                                                                   lower_percentile_rank,
                                                                                   upper_percentile_rank)
        new_raw = np.stack((new_red_component, new_green_component, new_blue_component), axis=2)
        return RGBImage(new_raw)

