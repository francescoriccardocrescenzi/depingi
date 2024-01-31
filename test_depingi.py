"""Unit tests for depingi.py."""
import unittest
import numpy as np
from depingi import LuminanceImage, RGBImage, RGBImageHistogram, LuminanceImageHistogram
import PIL.Image
import matplotlib.pyplot as plt


class TestLuminanceImage(unittest.TestCase):

    def test_constructor_file(self):
        im = RGBImage.from_file("data/plants.jpg")
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_constructor_from_raw_data(self):
        raw_data = np.asarray(PIL.Image.open("data/plants.jpg"))
        im = RGBImage(raw_data)
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_histogram(self):
        """Open a sample image, desaturate it and represent its histogram."""
        color_im = RGBImage.from_file("data/plants.jpg")
        gs_im = color_im.luminosity_desaturated()
        hist = gs_im.histogram()
        # Test whether the histograms and bins have the right dimensions.
        self.assertEqual(hist.values.shape, (gs_im.histogram_bin_number(),))
        self.assertEqual(hist.bins.shape, (gs_im.histogram_bin_number() + 1,))
        # Show the histogram
        # hist.show()

    def test_apply_contrast_stretching(self):
        """Initialize an RGB image from file, desaturate it and apply contrast stretching.

        Show the histogram of the picture before and after the desaturation.
        """

        def plot_histograms(histograms):
            figure, axes = plt.subplots(len(histograms))
            for index, hist in enumerate(histograms):
                axes[index].bar(hist.bins[:-1], hist.values, width=np.diff(hist.bins), edgecolor='black',
                                color='skyblue')
                axes[index].set_title(f"Histogram {index}")

        color_im = RGBImage.from_file("data/plants.jpg")
        gs_im1 = color_im.luminosity_desaturated()
        hist1 = gs_im1.histogram()
        gs_im2 = gs_im1.apply_contrast_stretching()
        hist2 = gs_im2.histogram()
        # plot_histograms([hist1, hist2])
        # gs_im1.show()
        # gs_im2.show()
        # plt.show()


class TestRGBImage(unittest.TestCase):

    def test_luminosity_desaturation(self):
        """Initialize a color image from file and desaturate it using the luminosity_desaturated method."""
        color_im = RGBImage.from_file("data/plants.jpg")
        gs_im = color_im.luminosity_desaturated()
        self.assertTrue(isinstance(gs_im, LuminanceImage))
        self.assertEqual(len(gs_im.raw.shape), 2)

    def test_apply_contrast_stretching(self):
        """Initialize an RGB image from file and apply contrast stretching.

        Show the histogram of the picture before and after the desaturation.
        """

        def plot_histograms(histograms):
            figure, axes = plt.subplots(len(histograms))
            for index, hist in enumerate(histograms):
                axes[index].bar(hist.bins[:-1], hist.values, width=np.diff(hist.bins), edgecolor='black', color='black')
                axes[index].set_title(f"Histogram {index}")

        im1 = RGBImage.from_file("data/plants.jpg")
        # hist1 = gs_im1.histogram()
        im2 = im1.apply_contrast_stretching(lower_percentile_rank=15, upper_percentile_rank=75)
        # hist2 = gs_im2.histogram()
        # plot_histograms([hist1, hist2])
        # im1.show()
        # im2.show()
        # plt.show()

    def test_histogram(self):
        """Plot the histogram of an RGB image."""
        def plot_histogram(histogram: RGBImageHistogram):
            figure, axes = plt.subplots()
            axes.bar(histogram.red_bins[:-1], hist.red_values, width=np.diff(hist.red_bins), edgecolor='red',
                        color='red')
            axes.set_title(f"Red")

            axes.bar(histogram.green_bins[:-1], hist.green_values, width=np.diff(hist.green_bins), edgecolor='green',
                        color='green')
            axes.set_title(f"Green")

            axes.bar(histogram.blue_bins[:-1], hist.blue_values, width=np.diff(hist.blue_bins), edgecolor='blue',
                        color='blue')
            axes.set_title(f"Red")

        im = RGBImage.from_file("data/plants.jpg")
        hist = im.histogram()
        plot_histogram(hist)
        plt.show()



if __name__ == "__main__":
    unittest.main()
