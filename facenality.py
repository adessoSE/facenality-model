import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator

# Import CNN averaged questionnaire results as output
cattells16_raw = pd.read_json("c:/users/grilborzer/facenality-db.json")
cattells16 = cattells16_raw.iloc[:, 0]

# Import photos with neutral expression as input
imagePath = 'c:/users/grilborzer/DecodedImages'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        imagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')