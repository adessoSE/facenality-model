# Imports

# Print library version

# Conda environment


# Create model
def create_model(image_size=IMAGE_SIZE):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(
        image_size, image_size, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    number_of_layers = 16

    for i in range(number_of_layers):
        model.add(Dense(units=2, kernel_initializer="uniform",
                        activation="relu", input_dim=x.shape[1]))
        if number_of_layers == 8:
            model.add(Dropout(0.2))

    model.add(Dense(16, activation="linear"))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

# Train model
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
        model.fit(X_train, y_train, epochs=500, batch_size=40)
        model.save_weights(WEIGHTS_NAME)

    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("Evaluation score: ", scores)

    return model