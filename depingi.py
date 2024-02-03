"""A handy Python module for image processing."""
import numpy as np


class Image:
    """Class used to open and process images.

    The image is stored in raw form as a float numpy ndarray whose values
    are comprised between 0 and 1.
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
        """Setter for self.raw.

        Check whether raw_data is of an accepted data type (integer or float). If raw_data is an integer array,
        normalize it. If it is a float array, leave it as is. If raw_data is not of an accepted type,
        raise a type error.
        """
        if np.issubdtype(raw_data.dtype, np.integer):
            raw_data = raw_data.astype(np.float32)/self.UINT8MAX
        elif np.issubdtype(raw_data.dtype, np.floating):
            pass
        else:
            raise TypeError("The numpy array used to initialize an instance of Image must be either a float array" +
                            " or an integer array.")
        self._raw = raw_data

    @property
    def raw_as_uint8(self):
        """Return self.raw as a uint8 array whose values are comprised between 0 and 255."""
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

    # DESATURATION

    def weighted_average_desaturated(self, weights: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])) -> LImage:
        """Convert to a luminance image using a weighted average of the RGB channels.

        The luminance of each pixel is determined by taking the weighted average of the red, green, and blue components
        according to the provided weights.

        Arguments:
            - weight: array in the form [wr, wg, wb], where wr, wg, wb are the weights used for the average.
        """
        # We use a tensor product to take the weighted average.
        desaturated_raw = np.tensordot(self.raw, weights, axes=(2, 0))
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


