from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional


def create_gru_model(series_len, num_appliances):
    """
    Creates a Gated Recurrent Unit model.

    Returns
    -------
    model : keras.models.Sequential
    """
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1,
                     input_shape=(series_len, 1)))
    # shape = (batch, series_len, 16)
    model.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))
    # shape = (batch, series_len, 8)

    # Bi-directional LSTMs
    model.add(Bidirectional(GRU(64, return_sequences=True, stateful=False),
                            merge_mode='concat'))
    # shape = (batch, series_len, 128)
    model.add(Bidirectional(GRU(128, return_sequences=True, stateful=False),
                            merge_mode='concat'))
    # shape = (batch, series_len, 256)

    # Fully Connected Layers
    model.add(Dense(num_appliances, activation='relu'))
    # shape = (batch, series_len, num_appliances)

    model.compile(loss='mse', optimizer='adam')

    return model
