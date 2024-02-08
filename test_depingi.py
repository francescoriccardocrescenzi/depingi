"""Unit tests for depingi.py."""
import unittest as ut
import depingi as dp
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt


def open_image(file_path):
    """Open image from file using PIL."""
    return np.asarray(PIL.Image.open(file_path))


def rgb_pil_image(raw_data):
    """Create PIL image from raw RGB data."""
    return PIL.Image.fromarray(raw_data)


def l_pil_image(raw_data):
    """Create PIL image from raw luminance data."""
    return PIL.Image.fromarray(raw_data, mode="L")


def plot_l_histograms(hists):
    """Plot a list of LImageHistogram instances."""
    fig, axes = plt.subplots(len(hists))
    # The default behavior of subplots is to only create a list of axes if len(hists) > 1.
    # If len(hists) == 1, encapsulate it into a list not to break the for loop below.
    if len(hists) == 1:
        axes = [axes]
    for i, hist in enumerate(hists):
        axes[i].bar(hist.bins[:-1], hist.values, width=np.diff(hist.bins), edgecolor="black")


def plot_rgb_histograms(hists):
    """Plot a list of RGBHistogram instances."""
    fig, axes = plt.subplots(len(hists))
    # The default behavior of subplots is to only create a list of axes if len(hists) > 1.
    # If len(hists) == 1, encapsulate it into a list not to break the for loop below.
    if len(hists) == 1:
        axes = [axes]
    for i, hist in enumerate(hists):
        axes[i].bar(hist.bins[0][:-1], hist.values[0], width=np.diff(hist.bins[0]), edgecolor="red")
        axes[i].bar(hist.bins[1][:-1], hist.values[1], width=np.diff(hist.bins[1]), edgecolor="green")
        axes[i].bar(hist.bins[2][:-1], hist.values[2], width=np.diff(hist.bins[2]), edgecolor="blue")


class TestImage(ut.TestCase):
    """Test Image class."""

    def test_getter_and_setter_for_raw(self):
        """Test getter and setter for self.raw."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        im = dp.Image(raw)
        uint8raw = im.raw_as_uint8
        pil_im = rgb_pil_image(uint8raw)
        # pil_im.show()


class TestRGBImage(ut.TestCase):
    """Test LImage class."""

    def test_desaturation_methods(self):
        """Open a colored image and desaturate it in three different ways."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        rgb_im = dp.RGBImage(raw)
        l_im1 = rgb_im.weighted_average_desaturated()
        l_im2 = rgb_im.luminosity_desaturated()
        l_im3 = rgb_im.lightness_desaturated()
        pil_im1 = l_pil_image(l_im1.raw_as_uint8)
        pil_im2 = l_pil_image(l_im2.raw_as_uint8)
        pil_im3 = l_pil_image(l_im3.raw_as_uint8)
        pil_im4 = rgb_pil_image(rgb_im.raw_as_uint8)
        # pil_im1.show()
        # pil_im2.show()
        # pil_im3.show()
        # pil_im4.show()

    def test_histogram(self):
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        im = dp.RGBImage(raw)
        hist = im.histogram()
        # plot_rgb_histograms([hist])
        # plt.show()

    def test_alpha_sum(self):
        """Take the alpha sum of two RGBImage instances."""
        path1 = "data/car.jpg"
        path2 = "data/plants.jpg"
        raw1 = open_image(path1)
        raw2 = open_image(path2)
        im1 = dp.RGBImage(raw1)
        im2 = dp.RGBImage(raw2)
        height = min(im1.raw.shape[0], im2.raw.shape[0])
        width = min(im1.raw.shape[1], im2.raw.shape[1])
        image_sum = dp.Image.alpha_sum(im1[:height, :width], im2[:height, :width], 0.9)
        pil_im = rgb_pil_image(image_sum.raw_as_uint8)
        # pil_im.show()

    def test_slicing(self):
        """Slice an RGBImage."""
        path = "data/car.jpg"
        raw = open_image(path)
        im = dp.RGBImage(raw)
        im = im[:im.raw.shape[0]//2, :im.raw.shape[1]//2]
        pil_im = rgb_pil_image(im.raw_as_uint8)
        # pil_im.show()

    def test_contrast_stretching(self):
        """Apply contrast stretching to an RGBImage."""
        file_path = "data/car.jpg"
        raw = open_image(file_path)
        rgb_im1 = dp.RGBImage(raw)
        rgb_im2 = rgb_im1.apply_contrast_stretching(15, 75)
        pil_im1 = rgb_pil_image(rgb_im1.raw_as_uint8)
        pil_im2 = rgb_pil_image(rgb_im2.raw_as_uint8)
        pil_im1.show()
        pil_im2.show()


class TestLImage(ut.TestCase):
    """Test RGBImage class."""

    def test_histogram(self):
        """Plot the histogram of an LImage."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        rgb_im = dp.RGBImage(raw)
        l_im = rgb_im.weighted_average_desaturated()
        hist = l_im.histogram()
        # plot_l_histograms([hist])
        # plt.show()

    def test_alpha_sum(self):
        """Take the alpha sum of two LImages."""
        path1 = "data/car.jpg"
        path2 = "data/plants.jpg"
        raw1 = open_image(path1)
        raw2 = open_image(path2)
        im1 = dp.RGBImage(raw1).lightness_desaturated()
        im2 = dp.RGBImage(raw2).lightness_desaturated()
        height = min(im1.raw.shape[0], im2.raw.shape[0])
        width = min(im1.raw.shape[1], im2.raw.shape[1])
        image_sum = dp.Image.alpha_sum(im1[:height, :width], im2[:height, :width], 0.1)
        pil_im = l_pil_image(image_sum.raw_as_uint8)
        # pil_im.show()

    def test_binary_image(self):
        """Take the binary image of an LImage."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        rgb_im = dp.RGBImage(raw)
        l_im = rgb_im.weighted_average_desaturated()
        bin_im = l_im.binary_image(.8)
        pil_im = l_pil_image(bin_im.raw_as_uint8)
        # pil_im.show()

    def test_contrast_stretching(self):
        """Apply contrast stretching to an LImage."""
        file_path = "data/plants.jpg"
        raw = open_image(file_path)
        rgb_im = dp.RGBImage(raw)
        l_im1 = rgb_im.weighted_average_desaturated()
        l_im2 = l_im1.apply_contrast_stretching(45, 50)
        pil_im1 = l_pil_image(l_im1.raw_as_uint8)
        pil_im2 = l_pil_image(l_im2.raw_as_uint8)
        # pil_im1.show()
        # pil_im2.show()


if __name__ == "__main__":
    ut.main()
