"""Unit tests for depingi.py."""
import unittest
import numpy as np
from depingi import LuminanceImage, RGBImage
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
        """Initialize and RGB image from file, desaturate it and apply contrast stretching.

        Show the histogram of the picture before and after the desaturation.
        """
        def plot_histograms(histograms):
            figure, axes = plt.subplots(len(histograms))
            for index, hist in enumerate(histograms):
                axes[index].bar(hist.bins[:-1], hist.values, width=np.diff(hist.bins), edgecolor='black', color='skyblue')
                axes[index].set_title(f"Histogram {index}")

        color_im = RGBImage.from_file("data/plants.jpg")
        gs_im1 = color_im.luminosity_desaturated()
        hist1 = gs_im1.histogram()
        gs_im2 = gs_im1.apply_contrast_stretching()
        hist2 = gs_im2.histogram()
        plot_histograms([hist1, hist2])
        gs_im1.show()
        gs_im2.show()
        plt.show()






class TestRGBImage(unittest.TestCase):

    def test_luminosity_desaturation(self):
        """Initialize a color image from file and desaturate it using the luminosity_desaturated method."""
        color_im = RGBImage.from_file("data/plants.jpg")
        gs_im = color_im.luminosity_desaturated()
        self.assertTrue(isinstance(gs_im, LuminanceImage))
        self.assertEqual(len(gs_im.raw.shape), 2)


if __name__ == "__main__":
    unittest.main()
