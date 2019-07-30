import constants as c

import numpy as np
from keras.preprocessing import image


def read_img(path):
    img = image.load_img(path, target_size=(c.IMAGE_SIZE, c.IMAGE_SIZE))
    return image.img_to_array(img)


def read_img_expand_dims(path):
    img = image.load_img(path, target_size=(c.IMAGE_SIZE, c.IMAGE_SIZE))
    img = np.expand_dims(img, axis = 0)
    return image.img_to_array(img)