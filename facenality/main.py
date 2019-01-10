# Imports
from facenality import constants as c
from facenality.preprocessing import data_import
from facenality.visualization import calculate_traits_mean

# Print library version

# Conda environment


def import_x_and_y_and_ids():
    json_path = c.DATA_SET + "/all.json"
    image_folder_path = c.DATA_SET + "/all/neutral/"

    y, y_id_list = data_import.import_y(json_path)
    x = data_import.import_x_as_np_array(image_folder_path, y_id_list)

    return x, y, y_id_list


if __name__ == '__main__':
    x, y, y_id_list = import_x_and_y_and_ids()
    y = y.tolist()

    y_mean = calculate_traits_mean(y)
    print(y_mean)