from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten
from keras.layers import Dense, ZeroPadding2D, Add, AveragePooling2D, MaxPool2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from models import ResNet18, CNN, DNN

def pixel_normalization(image):
    return (image / 255)


def get_data():
    dataset = mnist.load_data('mnist.db')
    train, test = dataset
    print(len(train))
    X_train, y_train = train
    X_test, y_test = test
    print(len(X_train))
    print(len(X_test))
    print(X_test.shape)

    X_train = pixel_normalization(X_train)
    X_test = pixel_normalization(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    print(X_test.shape)
    return X_train, y_train, X_test, y_test