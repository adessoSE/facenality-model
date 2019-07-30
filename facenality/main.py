import constants as c

from preprocessing import data_import as di
from model import model


def import_dataset():
    y, y_and_id = di.import_y(c.PATH_Y)
    x = di.import_x(c.PATH_X, y_and_id)

    return x, y, y_and_id


if __name__ == '__main__':
    x, y, y_and_id = import_dataset()
    y = y.tolist()

    model.Model.define_model_structure()