from . import image_utils as iu

import numpy as np
import pandas as pd


# Import image data x to corresponding list of given IDs
def import_x(folder_path, y_id_list):
    x = []

    for i in y_id_list:
        image_path = folder_path + str(i) + ".jpg"
        x.append(iu.read_img(image_path))

    x = np.array(x)
    return x


# Import questionnaire data y and a list of its IDs from a JSON file
def import_y(file_path_json):
    y_id_list = pd.read_json(file_path_json)
    y = y_id_list.iloc[:, 0].values
    y_id_list = y_id_list.id

    return y, y_id_list