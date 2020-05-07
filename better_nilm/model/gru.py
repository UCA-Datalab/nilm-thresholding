from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional

from better_nilm.model.loss import regression_loss
from better_nilm.model.loss import classification_loss


def create_gru_model(series_len, num_appliances):
    """
    Creates a Gated Recurrent Unit model.

    Returns
    -------
    model : keras.models.Sequential
    """

    # Input layer (batch, series_len, 1)
    inputs = Input(shape=(series_len, 1))

    # 1D Conv (batch, series_len, 16)
    conv1 = Conv1D(16, 4, activation="relu", padding="same", strides=1)(inputs)
    # 1D Conv (batch, series_len, 8)
    conv2 = Conv1D(8, 4, activation="relu", padding="same", strides=1)(conv1)

    # Bi-directional LSTMs (batch, series_len, 128)
    gru1 = Bidirectional(GRU(64, return_sequences=True, stateful=False),
                         merge_mode='concat')(conv2)
    # Bi-directional LSTMs (batch, series_len, 256)
    gru2 = Bidirectional(GRU(128, return_sequences=True, stateful=False),
                         merge_mode='concat')(gru1)

    # Fully Connected Layers (batch, series_len, num_appliances)
    dense = Dense(num_appliances, activation='relu')(gru2)

    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss=[regression_loss(), classification_loss()],
                  loss_weights=[1, 1],
                  optimizer='adam')

    return model
