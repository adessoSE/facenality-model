# Using CPU if commented
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd

import gc
from keras.backend.tensorflow_backend import set_session, clear_session, get_session

from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta
from sklearn.metrics import confusion_matrix, classification_report


LOAD_WEIGHTS = True
WEIGHTS_NAME = "facenality_weights2.h5"

# Reset Keras Session


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()

    # if it's done something you should see a number being outputted
    print("\nGarbage Collector: ", gc.collect())


def import_data():
    y_with_id = pd.read_json("dataset/all.json")
    y = y_with_id.iloc[:, 0].values
    return y, y_with_id


def load_train_data(y_with_id, image_size=224, image_path="dataset/all/neutral/"):
    x = []

    print('Read train images')
    for i in y_with_id.id:
        path = image_path + str(i) + ".jpg"
        x.append(read_img(path, image_size))

    x = np.array(x)
    return x


def read_img(path, image_size):
    img = image.load_img(path, target_size=(image_size, image_size))
    #img = np.expand_dims(img, axis = 0)
    return image.img_to_array(img)


def create_model(image_size=224):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(
        image_size, image_size, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    number_of_layers = 16

    for i in range(number_of_layers):
        model.add(Dense(units=14, kernel_initializer="uniform",
                        activation="relu", input_dim=x.shape[1]))

    model.add(Dense(16, activation="linear"))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model


def train_model(batch_size=30, nb_epoch=20):
    y, y_with_id = import_data()
    y = y.tolist()
    y = np.array(y)

    x = load_train_data(y_with_id)

    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=test_size, random_state=56741)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    model = create_model()

    if LOAD_WEIGHTS:
        model.load_weights(WEIGHTS_NAME)
    else:
        model.fit(X_train, y_train, epochs=100, batch_size=50)
        model.save_weights(WEIGHTS_NAME)

    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("Evaluation score: ", scores)

    return model


def predict(model, y):
    X_test = read_img("dataset/test/neutral/93.jpg", 224)
    X_test = np.expand_dims(X_test, axis=0)

    y_pred_detailed = model.predict(X_test)
    y_pred_detailed = y_pred_detailed[0]
    y_pred = []

    for i in y_pred_detailed:
        y_pred.append(round(i, 1))

    print("y_pred: ", y_pred)
    print("y_test: ", y[45])


if __name__ == "__main__":
    y, y_with_id = import_data()
    y = y.tolist()
    x = load_train_data(y_with_id)

    reset_keras()
    model = train_model()

    predict(model, y)
