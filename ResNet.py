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


def pixel_normalization(image):
    return (image / 255)


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def ResNet18(shape = (28, 28, 1), classes = 10):
    # Step 1 (Setup Input Layer)
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(2):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = AveragePooling2D((2,2), padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dense(classes, activation = 'softmax')(x)
    model = Model(inputs = x_input, outputs = x, name = "ResNet18")
    return model


dataset = mnist.load_data('mnist.db')
train,test = dataset
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

#'''
# Build the ResNet model
res = ResNet18()

# Compile the model
res.compile(optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=CategoricalAccuracy())
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

res.summary()

history = res.fit(X_train, y_train, epochs=10, validation_split=0.2)
test_loss, test_accuracy = res.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred = res.predict(X_test)
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
#'''