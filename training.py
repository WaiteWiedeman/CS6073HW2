from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def training(model, X_train, y_train, learn_rate, epochs):
    opt = Adam(learning_rate=learn_rate)

    model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_categorical_accuracy',
                                 save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    return history