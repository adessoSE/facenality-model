import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta


def import_data():
    y_with_id  = pd.read_json("dataset/all.json")
    y = y_with_id.iloc[:, 0].values
    return y, y_with_id


def load_train_data(y_with_id, image_size = 224, image_path = "dataset/all/neutral/"):   
    x = []
    
    print('Read train images')
    for i in y_with_id.id:
        path = image_path + str(i) + ".jpg"
        img = image.load_img(path, target_size = (image_size, image_size))
        img = image.img_to_array(img)
        #img = np.expand_dims(img, axis = 0)
        x.append(img)
    
    x = np.array(x)
    return x


def create_model(image_size = 224):
    model = Sequential()
            
    model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation= "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())
      
    model.add(Dense(units = 64, kernel_initializer = "uniform", activation = "relu", input_dim = x.shape[1]))
    model.add(Dense(1, activation= "linear"))
         
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model


def train_model(batch_size = 30, nb_epoch = 20):
    y, y_with_id = import_data()
    y = y.tolist()
    y = np.array(y)
    
    x = load_train_data(y_with_id)
    
    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=test_size, random_state=56741)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    model = create_model()
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, epochs = 10, batch_size = 30)
    
    return model

    
if __name__ == "__main__":
    y, y_with_id = import_data()
    y = y.tolist()
    
    a = np.array(y)
    x = load_train_data(y_with_id)
    #print(x)
    x1 = x[0]
    x1 = np.expand_dims(x1, axis = 0)
    
    train_model()