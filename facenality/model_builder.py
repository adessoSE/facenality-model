import constants as c
from preprocessing import data_import as di
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


def build_classification_models():
    models = []
    trained_models = []

    for i in range(16):
        models.append(create_simple_model())

    for i in range(16):
        trained_models.append(train_simple_model(model= models[i], trait_to_classify= c.TRAITS[i]))

    return models


def create_simple_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     input_shape=(c.IMAGE_SIZE, c.IMAGE_SIZE, c.COLOR_CHANNELS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    for i in range(c.HIDDEN_LAYERS):
        model.add(Dense(units=16, kernel_initializer="uniform",
                        activation="relu", input_shape=(c.IMAGE_SIZE, c.IMAGE_SIZE, c.COLOR_CHANNELS)))

    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model


def train_simple_model(model, trait_to_classify, batch_size = 16):
    data_train_base_directory = "../dataset/classification/train/"
    data_validation_base_directory = "../dataset/classification/validation/"

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            "../dataset/classification/train/" + trait_to_classify,  # this is the target directory
            target_size=(c.IMAGE_SIZE, c.IMAGE_SIZE),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    validation_generator = test_datagen.flow_from_directory(
            "../dataset/classification/validation/" + trait_to_classify,
            target_size=(c.IMAGE_SIZE, c.IMAGE_SIZE),
            batch_size=batch_size,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    model.save_model("../models/classification/" + "facenality-" + trait_to_classify + ".h5")
    return model
    

if __name__ == '__main__':
    build_classification_models()