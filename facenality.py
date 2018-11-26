import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split


class Facenality:
    # Image Array
    x = []
    
    # Cattells 16
    y = []
    y_with_id = []

    image_size = 224
        
    
    def __init__(self):
         self.import_data()


    def import_data(self):
        self.y_with_id  = pd.read_json("dataset/all.json")
        self.y = self.y_with_id.iloc[:, 0].values


    def load_train_data(self, image_path = "dataset/all/neutral/"):   
        print('Read train images')
        for i in self.y_with_id.id:
            path = image_path + str(i) + ".jpg"
            img = image.load_img(path, target_size = (self.image_size, self.image_size))
            img = image.img_to_array(img)
            #img = np.expand_dims(img, axis = 0)
    
            self.x.append(img)

        self.x = np.array(self.x)
        print(self.x)
    
    
    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), input_shape=(self.image_size, self.image_size, 3), activation= "relu"))
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        model.add(Flatten())
                
        """
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
    
      
        model.add(Dense(256, activation= "relu"))
        model.add(Dropout(0.5))
    
        model.add(Dense(128, activation= "relu"))
        model.add(Dropout(0.5))
        model.add(Flatten())
        """
        model.add(Dense(units = 64, kernel_initializer = "uniform", activation = "relu", input_dim = self.x.shape[1]))
        model.add(Dense(1, activation= "linear"))
     
        model.compile(loss='mean_squared_error', optimizer=Adadelta())
        return model
    
    
    def train_model(self, batch_size = 30, nb_epoch = 20):
        self.load_train_data()
    
        #X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=test_size, random_state=56741)
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2, random_state = 0)
    
        model = self.create_model()
        #model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
        model.fit(X_train, y_train, epochs = 10, batch_size = 30)
    
        model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
    
        return model


if __name__ == "__main__":
    app = Facenality()
    app.train_model()