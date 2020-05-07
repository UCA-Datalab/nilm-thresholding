from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional

from keras import backend as K
from keras.layers import Susbtract
from keras.activations import softmax


def create_gru_model(series_len, num_appliances, classification_thresholds,
                     regression_weight=1, classification_weight=1):
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

    # Regression output
    # Fully Connected Layers (batch, series_len, num_appliances)
    regression = Dense(num_appliances, activation='relu')(gru2)

    # Classification output
    thresh = K.constant(classification_thresholds)
    substract = Susbtract()([regression, thresh])
    classification = softmax(substract)

    model = Model(inputs=inputs, outputs=[regression, classification])
    model.compile(loss=["mean_squared_error", "binary_crossentropy"],
                  loss_weights=[regression_weight, classification_weight],
                  optimizer='adam')

    return model
