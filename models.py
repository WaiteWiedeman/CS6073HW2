# import packages
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten
from keras.layers import Dense, ZeroPadding2D, Add, AveragePooling2D, MaxPool2D, Dropout
from keras.models import Model, Sequential


# define function to create identity block for residual neural network
# takes model and filter size as input and returns the model
def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # layer 1
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # layer 2
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    # add residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


# define function to create convolutional block for residual neural network
# takes model and filter size as input and returns the model
def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # layer 1
    x = Conv2D(filter, (3,3), padding='same', strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # layer 2
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    # processing residue with conv(1,1)
    x_skip = Conv2D(filter, (1,1), strides=(2,2))(x_skip)
    # add residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


# define ResNet18 function to build residual network after ResNet18 architecture
# takes shape and number of classes as input and returns the model
def ResNet18(shape=(28, 28, 1), classes=10):
    # setup input layer
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)
    # initial convolutional layer along with maxPool
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # define size of sub-blocks and initial filter size
    block_layers = [2, 2, 2, 2]
    filter_size = 64
    # add the residual network blocks
    for i in range(2):  # only using two blocks for quicker training
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # one residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = AveragePooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x, name="ResNet18")
    return model


# define CNN funtion to create convolutional neural network model
# returns CNN model
def CNN():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu',
                   input_shape=(28, 28, 1)))
    model.add(Dropout(.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(Dropout(.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(units=10, activation='softmax'))
    return model


# define DNN function to create deep neural network
# returns DNN model
def DNN():
    model = Sequential()
    model.add(Dense(units=512, input_shape=(784,), activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model
