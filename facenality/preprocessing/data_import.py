# Imports
from facenality.preprocessing import image_utils as iu

import numpy as np
import pandas as pd

# Print library version

# Conda environment


# Import image data x
def import_x_as_np_array(folder_path, y_id_list):
    x = []

    for i in y_id_list:
        image_path = folder_path + str(i) + ".jpg"
        x.append(iu.read_img(image_path))

    x = np.array(x)
    return x


# Import questionnaire data y and a list of its ids
def import_y(file_path_json):
    y_id_list = pd.read_json(file_path_json)
    y = y_id_list.iloc[:, 0].values
    y_id_list = y_id_list.id

    return y, y_id_list