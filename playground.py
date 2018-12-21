#
# import numpy as np
# import pandas as pd
# from keras.preprocessing import image
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#
# IMAGE_SIZE = 224
#
# def read_img(path, image_size):
#     img = image.load_img(path, target_size=(image_size, image_size))
#     #img = np.expand_dims(img, axis = 0)
#     return image.img_to_array(img)
#
#
# def predict():
#     # As provided by ID 45
#     y_pred_detailed = [3.0724888, 3.4322248, 3.0071514, 3.1229835, 2.938174,  2.9367702, 3.0432177,
#                     3.132992,  3.0858815, 3.1110396, 3.1369367, 3.0602264, 2.857199,  3.45796,
#                     3.1602013, 2.9521356]
#     y_pred = []
#     y_test = [2.8, 2.7, 2.7, 2.8, 2.7, 2.7, 3.2,
#             2.6, 2.9, 3.1, 2.7, 2.9, 3.1, 3.1, 3.2, 2.7]
#
#     for i in y_pred_detailed:
#         y_pred.append(round(i, 1))
#
#     print("y_pred: ", y_pred)
#     print("y_test: ", y_test)
#
#     # Doesn't work yet:
#     #cm = confusion_matrix(y_test, y_pred)
#     # print(cm)
#     #print(accuracy_score(y_test, y_pred))
#
#
# def predict_batch(image_path="dataset/predict/"):
#     start = 179
#     end = 189
#
#     X_test = []
#
#     for i in range(start, end, 1):
#         if i == 183:
#             continue
#         path = image_path + str(i) + ".jpg"
#         image = read_img(path, IMAGE_SIZE)
#         X_test.append(image)
#
#     X_test = np.array(X_test)
#     y_test = pd.read_json("dataset/predict.json").iloc[:, 0].values
#     y_test = y_test.tolist()
#
#     print("X: ", X_test)
#     print("Y: ", y_test)

if __name__ == "__main__":
    from PIL import Image

    im = Image.open("dataset/1.jpg")
    width, height = im.size
    # (left, upper, right, lower)
    crop_rectangle = (int(round(width * 0.25)), 0, int(round(width * 0.75)), height)
    cropped_im = im.crop(crop_rectangle)

    cropped_im.show()



