from depingi import Image

# weights = np.array([1, 0, 0])


# color_image = Image.open("data/plants.jpg")
# print(type(color_image))
# raw_color_image = np.asarray(color_image)
# raw_bw_image = np.tensordot(raw_color_image, weights,axes=(2,0))
# bw_image = Image.fromarray(raw_bw_image)
# bw_image.show()

im = Image.from_file("data/plants.jpg")
PILim = im.as_PIL_Image()
PILim.show()





