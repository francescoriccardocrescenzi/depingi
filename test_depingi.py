"""Unit tests for depingi.py."""
import unittest
import numpy as np
from depingi import LuminanceImage, RGBImage
import PIL.Image
import matplotlib.pyplot as plt


class TestImage(unittest.TestCase):

    def test_RGB_constructor1(self):
        im = RGBImage.from_file("data/plants.jpg")
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_RGB_constructor2(self):
        raw_data = np.asarray(PIL.Image.open("data/plants.jpg"))
        im = RGBImage(raw_data)
        self.assertTrue(isinstance(im.raw, np.ndarray))

    def test_RGB_to_L_conversion(self):
        RGBim = RGBImage.from_file("data/plants.jpg")
        Lim = RGBim.luminosity_desaturated()
        self.assertTrue(isinstance(Lim, LuminanceImage))
        self.assertEqual(len(Lim.raw.shape), 2)

    def test_L_histogram(self):
        """Open a sample image, desaturate it and plot its intensity histogram using matplotlib."""
        RGBim = RGBImage.from_file("data/plants.jpg")
        Lim = RGBim.luminosity_desaturated()
        hist, bins = Lim.histogram()
        # Test whether the histograms and bins have the right dimensions.
        self.assertEqual(hist.shape, (Lim.histogram_bin_number(),))
        self.assertEqual(bins.shape, (Lim.histogram_bin_number()+1,))
        # Plot the histogram using matplotlib.
        plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black', color='skyblue')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()




if __name__ == "__main__":
    unittest.main()
