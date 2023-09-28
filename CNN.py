from keras.datasets import mnist
from keras.models import Sequential, save_model, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def pixel_normalization(image):
    return (image / 255)

dataset = mnist.load_data('mnist.db')
train,test = dataset
print(len(train))
X_train, y_train = train
X_test, y_test = test
print(len(X_train))
print(len(X_test))

X_train = pixel_normalization(X_train)
X_test = pixel_normalization(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu', input_shape=(28, 28, 1)))
cnn.add(Dropout(.2))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
cnn.add(Dropout(.2))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(.5))
cnn.add(Dense(units=10, activation='softmax'))

cnn.compile(optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=CategoricalAccuracy())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_categorical_accuracy', save_best_only=True)
history = cnn.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test), callbacks=[checkpoint, reduce_lr])

y_pred = cnn.predict(X_test)
predictions = []
for instance in y_pred:
    predictions.append(np.argmax(instance))

plt.figure(figsize=(7, 7))
plt.plot(history.history['loss'], color='blue', label='loss');
plt.plot(history.history['val_loss'], color='red', label='val_loss');
plt.legend();
plt.title('Loss vs Validation Loss');
plt.tight_layout()

plt.figure(figsize=(7, 7))
plt.plot(history.history['categorical_accuracy'], color='blue', label='accuracy');
plt.plot(history.history['val_categorical_accuracy'], color='red', label='val_accuracy');
plt.legend();
plt.title('Accuracy vs Validation Accuracy');
plt.tight_layout()

plt.show()