# Imports
import constants as c
from keras.preprocessing import image
# Print library version

# Conda environment


# Read single image
def read_img(path):
    img = image.load_img(path, target_size=(c.IMAGE_SIZE, c.IMAGE_SIZE))
    #img = np.expand_dims(img, axis = 0)
    return image.img_to_array(img)
