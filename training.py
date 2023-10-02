from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# defining training function to train models
# takes the model, training features and classes, learning rate, and number of epochs as input
# returns model history object
def training(model, X_train, y_train, learn_rate, epochs):
    opt = Adam(learning_rate=learn_rate)  # using adam optimizer
    # compile model for training
    model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    # reduce learning rate function to improve performance
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    # checkpoint function to save model weights
    #checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_accuracy',save_best_only=True)
    # fit model using training data
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=reduce_lr)

    return history