# Import necessary libraries
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from itertools import cycle


dataset = mnist.load_data('mnist.db')
train,test = dataset
print(len(train))
X_train, y_train = train
X_test, y_test = test
print(len(X_train))
print(len(X_test))
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1, 28*28)
y_train = to_categorical(y_train)
print(y_test.shape)

model=Sequential()
model.add(
    Dense(units = 512 , input_shape = (784,) , activation = 'relu' )
)
model.add(
    Dense(units = 256 , activation = 'relu' )
)
model.add(
    Dense(units = 128 , activation = 'relu' )
)
model.add(
    Dense(units = 64 , activation = 'relu' )
)
model.add(
    Dense(units = 10 , activation = 'softmax' )
)
model.summary()
model.compile(
    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)
history = model.fit(X_train, y_train, epochs=3, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(X_test, to_categorical(y_test))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_pred)
lb = LabelBinarizer().fit(y_test)
y_label = lb.transform(y_test)
n_classes = 10
fpr = []
tpr = []
thresholds = []
roc_auc = []
# Calculate ROC curve
#for i in range(n_classes):
    #fpr[i], tpr[i], thresholds[i] = roc_curve(y_label[:,i], y_pred[:,i])
    #roc_auc[i] = roc_auc_score(y_label[:,i], y_pred[:,i])

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(roc_auc)  # print(f"Test AUC Score: {roc_auc * 100:.2f}%")

fig, ax = plt.subplots(figsize=(6, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'fuchsia',
                'red', 'brown', 'green', 'cyan', 'gray', 'purple'])
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i, color in zip(range(n_classes),colors):
    RocCurveDisplay.from_predictions(
        y_label[:, i],
        y_pred[:, i],
        name=f"ROC curve for {target_names[i]}",
        color=color,
        ax=ax,
        plot_chance_level=(i == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
plt.legend()
plt.show()
