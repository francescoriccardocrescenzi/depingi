"""A handy Python module for image processing."""
import numpy as np
from abc import abstractmethod


# IMAGE HISTOGRAMS

class ImageHistogram:
    """Class used to store and display the histograms for Image instances."""
    pass


class LImageHistogram(ImageHistogram):
    """Helper class used to store and display the histograms for LImage instances."""

    def __init__(self, values: np.ndarray, bins: np.ndarray):
        """Initialize a new instance of LImageHistogram.

        Arguments:
            - values: histogram values
            - bins: histogram bins
        """
        self.values = values
        self.bins = bins


class RGBImageHistogram(ImageHistogram):
    """Helper class used to store and display the histograms for RGBImage instances."""

    def __init__(self, values: list, bins: list):
        """Initialize a new instance of RGImageHistogram.

        Arguments:
            - values: list of histogram values.
                values[0] -> values associated with red component
                values[1] -> values associated with green component
                values[2] -> values associated with blue component
            - bins: list of histogram bins edges.
                bins[0] -> bins associated with red component
                bins[1] -> bins associated with green component
                bins[2] -> bins associated with blue component
        """
        self.values = values
        self.bins = bins


# IMAGES

class Image:
    """Class used to open and process images.

    The image is stored in raw form as a float numpy ndarray whose values
    are comprised between 0 and 1.
    """

    # CONSTANTS

    # Alias the uint8 maximum and minimum values for increased legibility.
    _UINT8MAX = np.iinfo(np.uint8).max
    _UINT8MIN = np.iinfo(np.uint8).min

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
        """Setter for self.raw.

        Check whether raw_data is of an accepted data type (integer or float). If raw_data is an integer array,
        normalize it. If it is a float array, leave it as is. If raw_data is not of an accepted type,
        raise a type error.
        """
        if np.issubdtype(raw_data.dtype, np.integer):
            raw_data = raw_data.astype(np.float32) / self._UINT8MAX
        elif np.issubdtype(raw_data.dtype, np.floating):
            pass
        else:
            raise TypeError("The numpy array used to initialize an instance of Image must be either a float array" +
                            " or an integer array.")
        self._raw = raw_data

    @property
    def raw_as_uint8(self) -> np.ndarray:
        """Return self.raw as a uint8 array whose values are comprised between 0 and 255."""
        return (self.raw * self._UINT8MAX).astype(np.uint8)

    # HISTOGRAM

    @property
    def _histogram_bin_number(self) -> int:
        """Helper method intended for internal used only.

        This method provides the number of bins of the histogram of the image.
        The default bin number is set to sqrt(n_pix), where npix is the number of pixels of the image.
        """
        n_pix = self.raw.shape[0] * self.raw.shape[1]
        return int(np.sqrt(n_pix))

    @abstractmethod
    def histogram(self, bin_number: int = None) -> ImageHistogram:
        """Returns the histogram of the image.

        Arguments:
            - bin_number: the number of bins that the histogram will have; if set to None,
                sqrt(self.raw.shape[0]*self.raw.shape[1]) will be used instead.
        """
        pass

    # IMAGE ARITHMETICS

    @classmethod
    def alpha_sum(cls, image1: "Image", image2: "Image", alpha: float) -> "Image":
        """Return the alpha sum of the two images according to the formula new_raw = alpha*raw1 + (1-alpha)*raw2.
        Arguments:
            - image1 and image2 are supposed to be Image instances of the same subclass,
                and they are supposed to have the same size.
            - alpha: float in the interval [0,1].
        Returns:
            - an Image of the same subclass of image1 and image2
        """
        new_raw = alpha * image1.raw + (1 - alpha) * image2.raw
        return cls(new_raw)

    # SLICING

    def __getitem__(self, item):
        """An image is sliced by slicing its raw data according to numpy conventions.

        Eg: if im is an image of subclass LImage, im[item] == LImage(im.raw[item]).
        """
        return type(self)(self.raw[item])

    # CONTRAST STRETCHING

    @classmethod
    def _apply_contrast_stretching_to_raw_luminance_image(cls, raw: np.ndarray,
                                                          lower_percentile_rank: int,
                                                          upper_percentile_rank: int) -> np.ndarray:
        """Helper method intended for internal use only.

        The method applies contrast stretching to a raw luminance image encoded as a numpy float array.
        If lower_percentile < upper percentile, the following formula is applied to each pixel:
        P_new = (P_old - lower_percentile)/(upper_percentile - lower_percentile).
        If lower_percentile >= upper percentile, the raw image is returned without change.

        Arguments:
            - raw: the raw image on which to apply contrast stretching
            - lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            - upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both percentile ranks should be integers between 1 and 99.
        """
        lower_percentile = np.percentile(raw, lower_percentile_rank)
        upper_percentile = np.percentile(raw, upper_percentile_rank)
        if lower_percentile < upper_percentile:
            new_raw = (raw - lower_percentile) / (upper_percentile - lower_percentile)
        else:
            new_raw = raw
        # Since we are using percentiles in place of maximum and minimum intensity, we need to clip the new raw image
        # before returning it.
        np.clip(new_raw, 0, 1, out=new_raw)
        return new_raw

    # HISTOGRAM EQUALIZATION

    @classmethod
    def _equalize_histogram_of_raw_luminance_image(cls, raw: np.ndarray, bin_number: int) -> np.ndarray:
        """Helper method meant for internal use only.

        Return a new raw luminance histogram with constant histogram.

        Arguments:
            - bin_number: number of bins used to compute the histogram.
        """
        # Take the histogram of raw, compute its cdf, normalize it and interpolate
        # the new raw from the normalized cdf.
        hist, bins = np.histogram(raw, bins=bin_number, range=(0, 1))
        cdf = hist.cumsum()
        new_cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        new_raw = np.interp(raw, bins[:-1], new_cdf)
        return new_raw


class LImage(Image):
    """Subclass of Image to open and process luminance images.

    The images are stored as 2d numpy arrays and the luminance of each pixel is represented as a float 0 < lum < 1.
    """

    # HISTOGRAM

    def histogram(self, bin_number: int = None) -> LImageHistogram:
        if bin_number is None:
            bin_number = self._histogram_bin_number
        values, bins = np.histogram(self.raw, bins=bin_number, range=(0, 1))
        return LImageHistogram(values, bins)

    # THRESHOLDING

    def binary_image(self, t: float) -> "LImage":
        """Return the binary image associated with the threshold t.

        new_P = 0 if old_P < t
        new_P = 1 if old_p >= t

        Arguments:
            - t: float in the interval [0,1].
        """
        binary_raw = (self.raw >= t).astype(np.float32)
        return LImage(binary_raw)

    # CONTRAST STRETCHING

    def apply_contrast_stretching(self, lower_percentile_rank: int = 1, upper_percentile_rank: int = 99) \
            -> "LImage":
        """
        Create a new LImage by applying contrast stretching.

        If lower_percentile < upper percentile, the following formula is applied to each pixel:
        P_new = (P_old - lower_percentile)/(upper_percentile - lower_percentile).
        If lower_percentile >= upper percentile, the raw image is returned without change.

        Arguments:
            -- lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            -- upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both percentiles should be integers between 1 and 99.
        """
        return LImage(Image._apply_contrast_stretching_to_raw_luminance_image(self.raw,
                                                                              lower_percentile_rank,
                                                                              upper_percentile_rank))

    # HISTOGRAM EQUALIZATION

    def equalize_histogram(self, bin_number: int = None):
        """Return a new LImage with an equalized histogram.

        Arguments:
            - bin_number: number of bins used to compute the histogram for the equalization; if no
                bin_number is inserted, the square root of the number of pixels of the image is used.
        """
        if bin_number is None:
            bin_number = self._histogram_bin_number
        return LImage(self._equalize_histogram_of_raw_luminance_image(self.raw, bin_number))


class RGBImage(Image):
    """Subclass of Image to open and process RGB images.

    The images are stored as 3d nxmx3 numpy arrays.
    """

    # DESATURATION

    def weighted_average_desaturated(self, weights: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])) -> LImage:
        """Convert to a luminance image using a weighted average of the RGB channels.

        The luminance of each pixel is determined by taking the weighted average of the red, green, and blue components
        according to the provided weights.

        Arguments:
            - weight: array in the form [wr, wg, wb], where wr, wg, wb are the weights used for the average.
        """
        desaturated_raw = sum([weights[i] * self.raw[:, :, i] for i in range(3)])
        return LImage(desaturated_raw)

    def luminosity_desaturated(self) -> LImage:
        """Convert to a luminance image using the luminosity method.

        Apply RGBImage.weighted_average_desaturated with weights [0.21, 0.72, 0.07].
        """
        return self.weighted_average_desaturated(np.array([0.21, 0.72, 0.07]))

    def lightness_desaturated(self) -> LImage:
        """Convert to a luminance image using the lightness method.

        The luminance of each pixel is determined as (min(R,G,B) + max(R,G,B))/2.
        """
        desaturated_raw = (np.amax(self.raw, axis=2) + np.amin(self.raw, axis=2)) / 2
        return LImage(desaturated_raw)

    # HISTOGRAM

    def histogram(self, bin_number: int = None) -> RGBImageHistogram:
        if bin_number is None:
            bin_number = self._histogram_bin_number
        values = []
        bins = []
        # Generate the red, green and blue histograms.
        # values[0] = red values, values[1] = green values, values[2] = blue values
        # bins[0] = red values, bins[1] = green values, bins[2] = blue values
        for i in range(3):
            v, b = np.histogram(self.raw[:, :, i], bins=bin_number, range=(0, 1))
            values.append(v)
            bins.append(b)
        return RGBImageHistogram(values, bins)

    # CONTRAST STRETCHING

    def apply_contrast_stretching(self, lower_percentile_rank: int = 1, upper_percentile_rank: int = 99) \
            -> "RGBImage":
        """
        Create a new RGBImage by applying contrast stretching.

        Apply to each channel of each pixel the following mathematical formula:
        P_new = ((P_old - lower_percentile)/(upper_percentile - lower_percentile))*255.
        The percentile ranks are the same for each channel but the actual percentiles depend on the specific channel.
        If lower_percentile >= upper_percentile for some channel, the formula is not applied to that channel
        and thus such channel is left unchanged.

        Arguments:
            -- lower_percentile_rank: percentile_rank of the lower percentile in the formula above
            -- upper_percentile_rank: percentile rank of the upper percentile in the formula above
        Both percentile ranks should be integers between 1 and 99.
        """
        # Apply the transformation to each channel using a helper method and stack the channels together to obtain a
        # well-formatted new raw image.
        components = [
            self._apply_contrast_stretching_to_raw_luminance_image(self.raw[:, :, i],
                                                                   lower_percentile_rank,
                                                                   upper_percentile_rank)
            for i in range(3)
        ]
        new_raw = np.stack(components, axis=2)
        return RGBImage(new_raw)

    # HISTOGRAM EQUALIZATION

    def equalize_histogram(self, bin_number: int = None):
        """Return a new RGBImage with an equalized histogram (across all channels).

        Arguments:
            - bin_number: number of bins used to compute the histogram for the equalization; if no
                bin_number is inserted, the square root of the number of pixels of the image is used.
        """
        if bin_number is None:
            bin_number = self._histogram_bin_number
        components = [
            self._equalize_histogram_of_raw_luminance_image(self.raw[:, :, i], bin_number)
            for i in range(3)
        ]
        new_raw = np.stack(components, axis=2)
        return RGBImage(new_raw)
