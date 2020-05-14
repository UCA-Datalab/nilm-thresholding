from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional
    

def create_gru_model(series_len, num_appliances,
                     regression_weight=1, classification_weight=1):
    """
    Creates a Gated Recurrent Unit model.
    Based on OdysseasKr GRU model:
    https://github.com/OdysseasKr/neural-disaggregator/blob/master/GRU

    Parameters
    ----------
    series_len : int
    num_appliances : int
    regression_weight : float, default=1
        Weight for the regression loss (MSE)
    classification_weight : float, default=1
        Weight for the classification loss (BCE)

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

    # Dense layer
    dense = Dense(64, activation='relu')(gru2)

    # Regression output
    # Fully Connected Layers (batch, series_len, num_appliances)
    regression = Dense(num_appliances, activation='relu',
                       name='regression')(dense)

    # Classification output
    # Fully Connected Layers (batch, series_len, num_appliances)
    classification = Dense(num_appliances, activation="sigmoid",
                           name="classification")(dense)

    model = Model(inputs=inputs,
                  outputs=[regression, classification])
    model.compile(loss={"regression": "mean_squared_error",
                        "classification": "binary_crossentropy"},
                  loss_weights={"regression": regression_weight,
                                "classification": classification_weight},
                  optimizer='adam')

    return model
