from keras.callbacks import EarlyStopping


def train_with_validation(model, x_train, y_train, x_val, y_val,
                          epochs=4000, batch_size=64, shuffle=True,
                          patience=200):
    """
    Train a model, implementing early stop. The train stops when the
    validation loss ceases to decrease.

    Parameters
    ----------
    model : keras.models.Model
    x_train : numpy.array
    y_train : numpy.array or list of numpy.array
    x_val : numpy.array
    y_val : numpy.array or list of numpy.array
    epochs : int, default=4000
        Number of epochs to train the model. An epoch is an iteration over
        the entire x and y data provided.
    batch_size : int, default=64
        Number of samples per gradient update.
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.
    patience : int, default=200
         Number of epochs with no improvement after which training will be
         stopped.

    Returns
    -------
    model : keras.models.Model

    """
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=patience)

    # Fit model
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, shuffle=shuffle,
              callbacks=[es])
    return model
