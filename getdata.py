# import packages
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


# define function to normalize images
# returns normalized image
def pixel_normalization(image):
    return (image / 255)


# define get data function
# returns training/testing features and classes
def get_data():
    dataset = mnist.load_data('mnist.db')  # load mnist dataset
    train, test = dataset  # split dataset into training and testing data
    X_train, y_train = train  # split training data into features and classes
    X_test, y_test = test  # split testing data into features and classes

    X_train = pixel_normalization(X_train)  # normalize training and testing features
    X_test = pixel_normalization(X_test)
    y_train = to_categorical(y_train)  # convert class to binary
    y_test = to_categorical(y_test)
    X_train = np.expand_dims(X_train, axis=3)  # change feature dimensions
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, y_train, X_test, y_test