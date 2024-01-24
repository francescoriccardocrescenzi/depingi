from depingi import LuminanceImage, RGBImage
import numpy as np

weights = np.array([1, 0, 0])

RGBim = RGBImage.from_file("data/plants.jpg")
Lim1 = RGBim.as_LuminanceImage()
Lim2 = RGBim.as_LuminanceImage(weights)
Lim1.as_PILImage().show()
Lim2.as_PILImage().show()




