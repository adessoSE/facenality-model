# Imports

# Print library version

# Conda environment


# Predict single image
def predict(model, y):
    X_test = read_img("dataset/test/neutral/93.jpg", IMAGE_SIZE)
    X_test = np.expand_dims(X_test, axis=0)

    y_pred_detailed = model.predict(X_test)
    y_pred_detailed = y_pred_detailed[0]
    y_pred = []

    for i in y_pred_detailed:
        y_pred.append(round(i, 1))

    print("y_pred: ", y_pred)
    print("y_test: ", y[45])

# Predict batch of images
def predict_batch(model, image_path="dataset/predict/"):
    # READ DATA
    start = 179
    end = 189

    X_test = []

    for i in range(start, end, 1):
        if i == 183:
            continue
        path = image_path + str(i) + ".jpg"
        image = read_img(path, IMAGE_SIZE)
        X_test.append(image)

    X_test = np.array(X_test)
    y_test_id = pd.read_json("dataset/predict.json")
    y_test = pd.read_json("dataset/predict.json").iloc[:, 0].values
    y_test = y_test.tolist()

    # PREDICT DATA
    y_pred_detailed = model.predict(X_test)
    y_pred = []
    n = 0

    for array in y_pred_detailed:
        temp = []
        for i in array:
            temp.append(round(i, 1))
        y_pred.append(temp)

        print("\n")
        print("ID: ", y_test_id.iloc[n].id)
        print("y_test", n, " : ", y_test[n])
        print("y_pred", n, " : ", temp)
        print("\n")
        n += 1
