from depingi import LuminanceImage, RGBImage
import numpy as np

col_im = RGBImage.from_file("data/plants.jpg")
gs_image = col_im.luminosity_desaturated()
gs_image.as_PILImage().show()

gs_image.apply_contrast_stretching()
gs_image.apply_contrast_stretching()
gs_image.apply_contrast_stretching()
gs_image.apply_contrast_stretching()
gs_image.apply_contrast_stretching()
gs_image.as_PILImage().show()





