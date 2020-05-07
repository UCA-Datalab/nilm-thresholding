from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional

from keras import backend as K
from keras.layers import Lambda
from keras.layers import Softmax


def _subtract_tensor(classification_thresholds):
    thresh = K.constant(classification_thresholds)
    def _lambda(x):
        return x - thresh
    return _lambda
    

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
    regression = Dense(num_appliances, activation='relu',
                       name='regression')(gru2)

    # Classification output
    substract = Lambda(_subtract_tensor(classification_thresholds))(regression)
    classification = Softmax(name='classification')(substract)

    model = Model(inputs=inputs,
                  outputs=[regression, classification])
    model.compile(loss={"regression": "mean_squared_error",
                        "classification": "binary_crossentropy"},
                  loss_weights={"regression": regression_weight,
                                "classification": classification_weight},
                  optimizer='adam')

    return model
