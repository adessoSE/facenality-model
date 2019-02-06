# Using GPU if commented
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd

import gc
from keras.backend.tensorflow_backend import set_session, clear_session, get_session

from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta, Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
    

LOAD_WEIGHTS = False

EXPRESSION = "neutral"
DATE = "06-02-2019"
DESCRIPTION = "vgg16-1"
WEIGHTS_NAME = "facenality-weights-" + EXPRESSION + "-" + DATE + "-" + DESCRIPTION + ".h5"
MODEL_NAME = "facenality-model-" + EXPRESSION + "-" + DATE + "-" + DESCRIPTION +".h5"

IMAGE_SIZE = 224
HIDDEN_LAYERS = 5
BATCH_SIZE = 32
EPOCHS = 16


# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()

    # if it's done something you should see a number being outputted
    print("\nGarbage Collector: ", gc.collect())


def import_data(path):
    y_with_id = pd.read_json(path)
    y = y_with_id.iloc[:, 0].values
    return y, y_with_id


def load_train_data(y_with_id, image_path, image_size=IMAGE_SIZE):
    x = []

    for i in y_with_id.id:
        path = image_path + str(i) + ".jpg"
        x.append(read_img(path, image_size))

    x = np.array(x)
    return x


def read_img(path, image_size=IMAGE_SIZE):
    img = image.load_img(path, target_size=(image_size, image_size))
    #img = np.expand_dims(img, axis = 0)
    return image.img_to_array(img)


def create_model():
    model = Sequential()
 
    model.add(Conv2D(32, (3, 3),
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())


    for i in range(HIDDEN_LAYERS):
        model.add(Dense(units=16, kernel_initializer="uniform",
                        activation="relu", input_dim=x.shape[1]))
        if HIDDEN_LAYERS == 8:
            model.add(Dropout(0.2))

    model.add(Dense(16, activation="linear"))

    #model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model


def train_model():
    y, y_with_id = import_data("../dataset/all.json")
    y = y.tolist()
    y = np.array(y)

    x = load_train_data(y_with_id, image_path = "../dataset/all-cropped/" + EXPRESSION + "/")

    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=test_size, random_state=56741)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    model = create_model()

    if LOAD_WEIGHTS:
        model.load_weights(WEIGHTS_NAME)
    else:
        model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)
        model.save_weights(WEIGHTS_NAME)
        model.save(MODEL_NAME)

    # Save the model
    #model.save(MODEL_NAME)

    # evaluate the model
    scores_train = model.evaluate(X_train, y_train)
    scores_test = model.evaluate(X_test, y_test)
    print("Evaluation score Train: ", scores_train)
    print("Evaluation score Test: ", scores_test)

    return model


def predict(model, y, image_path):
    X_test = read_img(image_path, IMAGE_SIZE)
    X_test = np.expand_dims(X_test, axis=0)

    y_pred_detailed = model.predict(X_test)
    y_pred_detailed = y_pred_detailed[0]
    y_pred = []

    for i in y_pred_detailed:
        y_pred.append(round(i, 1))

    print("y_pred: ", y_pred)
    print("y_test: ", y[45])


# Vorher nicht gecroppte Bilder verwendet
def predict_batch(model, image_path, shouldPrint = False):
    # READ DATA
    y_test_id = pd.read_json("../dataset/predict.json")
    y_test = pd.read_json("../dataset/predict.json").iloc[:, 0].values
    
    X_test = []

    for i in y_test_id.id:
        #if i == 183:
        #    continue
        path = image_path + str(i) + ".jpg"
        image = read_img(path, IMAGE_SIZE)
        X_test.append(image)

    X_test = np.array(X_test)

    y_test = y_test.tolist()

    # PREDICT DATA
    rmse_average = []
    y_pred_detailed = model.predict(X_test)
    y_pred = []
    n = 0

    if(shouldPrint):
        print("\n")
        print(image_path)
    for array in y_pred_detailed:
        temp = [] 
        for i in array:
            temp.append(round(i, 1))
        y_pred.append(temp)
        
        temp_np = np.array(temp)
        y_test_np = np.array(y_test[n])
        
        if(shouldPrint):
            print("ID: ", y_test_id.iloc[n].id)
            print("y_test", n, " : ", y_test[n])
            print("y_pred", n, " : ", temp)
        rmse_average.append(calculateRMSE(temp_np, y_test_np))
        
        n += 1
    
    rmse_average_np = np.array(rmse_average)
    rmse_average_np = np.average(rmse_average_np)
    if(shouldPrint):
        print("RMSE average for " + image_path + " : " + str(rmse_average_np))
    #scores_pred = model.evaluate(np.array(y_test), np.array(y_pred_detailed))
    #print("Evaluation score Prediction: ", scores_pred)
  
    
    #import keract
    #activations = keract.get_activations(model, X_test)
    #print(activations)
    #keract.display_activations(activations["conv2d_1/Relu:0"])
    #keract.display_activations(activations)
    return rmse_average_np
    

def calculateRMSE(y_pred, y_test, shouldPrint = False):
    lin_mse = mean_squared_error(y_pred, y_test)
    lin_rmse = np.sqrt(lin_mse)
    if(shouldPrint):
        print('Linear Regression RMSE: %.4f' % lin_rmse)
    return lin_rmse

    
def printModel():
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    from pptx_util import save_model_to_pptx
    #from keras_util import convert_drawer_model    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=16, kernel_initializer="uniform", activation="relu", input_dim=x.shape[1]))
    model.add(Dense(16, activation="linear"))
    
    #seqmodel = convert_drawer_model(model)

# save as svg file
    #model.save_fig("example.svg")
    save_model_to_pptx(model, "example.pptx")
    
    
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


def calculateAverage(shouldPrint = False):
    y, y_with_id = import_data("../dataset/all.json")
    y = y.tolist()

    y2, y2_with_id = import_data("../dataset/predict.json")
    y2 = y2.tolist()
    
    y_gesamt = y + y2
    y_avg = np.average(y_gesamt, axis = 0)
    
    if(shouldPrint):
        np.set_printoptions(precision=1)
        print("y avg as np array: ", y_avg)
    
    return y_avg


def calculateRMSE_of_validation_and_average_values(averageValues, validationSet):
    rmse_average = []
    n = 0

    print("\nRMSE of validation and average values")
    for validation_array in validationSet.cattells16Questions:
        
        print("ID: ", validationSet.id[n])
        print("validation", n, " : ", validation_array)
        print("   average  ", " : ", averageValues)
        rmse_average.append(calculateRMSE(np.array(averageValues), np.array(validation_array)))
        print("\n")
        n += 1
    
    rmse_average_np = np.array(rmse_average)
    rmse_average_np = np.average(rmse_average_np)
    print("RMSE average for validation and average values: ", rmse_average_np)
    


if __name__ == "__main__":
    y, y_with_id = import_data("../dataset/all.json")
    y = y.tolist()
    x = load_train_data(y_with_id, image_path = "../dataset/all-cropped/" + EXPRESSION + "/")

    #y_validation, y_validation_with_id = import_data("../dataset/predict.json")
    #y_validation = y_validation.tolist()
    #y_avg = calculateAverage()
    
    #calculateRMSE_of_validation_and_average_values(y_avg, y_validation_with_id)
    
    #reset_keras()
    model = train_model()
    model.summary()
        
    # 29.01 save again for test
    #model.save(MODEL_NAME)
    #printModel()
    
    rmse_average = []
    rmse_average_cropped = []
    
    rmse_average.append(predict_batch(model, "../dataset/predict/neutral/"))
    rmse_average_cropped.append(predict_batch(model, "../dataset/predict-cropped/neutral/", shouldPrint = True))

    rmse_average.append(predict_batch(model, "../dataset/predict/happy/"))
    rmse_average_cropped.append(predict_batch(model, "../dataset/predict-cropped/happy/", shouldPrint = True))

    rmse_average.append(predict_batch(model, "../dataset/predict/sad/"))
    rmse_average_cropped.append(predict_batch(model, "../dataset/predict-cropped/sad/", shouldPrint = True))
    
    rmse_average.append(predict_batch(model, "../dataset/predict/random/"))
    rmse_average_cropped.append(predict_batch(model, "../dataset/predict-cropped/random/", shouldPrint = True))
    
    print("Hidden layers: ", HIDDEN_LAYERS)
    print("\nRMSE averages for uncropped predictions with:    " + MODEL_NAME)
    print("neutral, happy, sad, random")
    print(str(rmse_average) + "\n")
    
    print("RMSE averages for cropped predictions with:    " + MODEL_NAME)
    print("neutral, happy, sad, random")
    print(str(rmse_average_cropped) + "\n")
    
