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
from getdata import get_data
from training import training
from testing import test, get_plot


learn_rate = 0.001
epochs = 10

data = get_data()

X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]

res = ResNet18()
cnn = CNN()
dnn = DNN()

res_history = training(res, X_train, y_train, learn_rate, epochs)
res_test_loss, res_test_accuracy = test(res, X_test, y_test)
get_plot(res_history)

cnn_history = training(cnn, X_train, y_train, learn_rate, epochs)
cnn_test_loss, cnn_test_accuracy = test(cnn, X_test, y_test)
get_plot(cnn_history)

dnn_history = training(dnn, X_train, y_train, learn_rate, epochs)
dnn_test_loss, dnn_test_accuracy = test(dnn, X_test, y_test)
get_plot(dnn_history)
