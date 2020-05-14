from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Activation
from keras.activations import sigmoid

from keras.optimizers import Adam
    

def create_gru_model(series_len, num_appliances, thresholds,
                     regression_weight=1, classification_weight=1,
                     learning_rate=0.001):
    """
    Creates a Gated Recurrent Unit model.
    Based on OdysseasKr GRU model:
    https://github.com/OdysseasKr/neural-disaggregator/blob/master/GRU

    Parameters
    ----------
    series_len : int
    num_appliances : int
    thresholds : numpy.array
        shape = (num_appliances, )
        Load threshold for each appliance
    regression_weight : float, default=1
        Weight for the regression loss (MSE)
    classification_weight : float, default=1
        Weight for the classification loss (BCE)
    learning_rate : float, default=0.001
        Starting learning rate for the Adam optimizer.

    Returns
    -------
    model : keras.models.Model

    """
    assert len(thresholds) == num_appliances, "Number of thresholds must " \
                                              "equal the amount of appliances"

    # ARCHITECTURE

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
    subtract = Lambda(lambda x: x - thresholds)(regression)
    # Fully Connected Layers (batch, series_len, num_appliances)
    classification = Activation(sigmoid, name='classification')(subtract)

    # TRAINING

    # Weights
    # We scale the weights because BCE grows bigger than MSE
    class_w = classification_weight * .003
    reg_w = regression_weight * .997

    # Optimizer
    opt = Adam(learning_rate=learning_rate)

    model = Model(inputs=inputs,
                  outputs=[regression, classification])
    model.compile(loss={"regression": "mean_squared_error",
                        "classification": "binary_crossentropy"},
                  loss_weights={"regression": reg_w,
                                "classification": class_w},
                  optimizer=opt)

    return model
