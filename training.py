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


def training(model, X_train, y_train, learn_rate, epochs):
    opt = Adam(learning_rate=learn_rate)

    model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    return history