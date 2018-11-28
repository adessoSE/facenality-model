from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# As provided by ID 45
y_pred_detailed = [3.0724888, 3.4322248, 3.0071514, 3.1229835, 2.938174,  2.9367702, 3.0432177,
                   3.132992,  3.0858815, 3.1110396, 3.1369367, 3.0602264, 2.857199,  3.45796,
                   3.1602013, 2.9521356]
y_pred = []
y_test = [2.8, 2.7, 2.7, 2.8, 2.7, 2.7, 3.2,
          2.6, 2.9, 3.1, 2.7, 2.9, 3.1, 3.1, 3.2, 2.7]

for i in y_pred_detailed:
    y_pred.append(round(i, 1))

print("y_pred: ", y_pred)
print("y_test: ", y_test)

# Doesn't work yet:
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#print(accuracy_score(y_test, y_pred))