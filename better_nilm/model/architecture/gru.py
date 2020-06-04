from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Dropout

from keras.backend import constant
from keras.layers import Lambda
from keras.layers import Activation
from keras.activations import sigmoid

from keras.optimizers import Adam

from better_nilm.model.architecture._base import KerasModel


class GRUModel(KerasModel):
    def __init__(self, series_len, num_appliances, thresholds,
                 regression_weight=1, classification_weight=1,
                 learning_rate=0.001, sigma_c=50, dropout=0.5):

        """
        Initializes a Gated Recurrent Unit model.
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
        sigma_c : float, default=50
            Controls the slope of the sigma function. Being T the threshold and
            C this parameters sigma_c, we define the sigma as:
            f(x) = ( 1 + exp( -C * (x - T) ) ) ^ (-1)
        dropout : float, default=0.5
            Dropout between layers.

        """

        super(KerasModel, self).__init__()

        assert len(thresholds) == num_appliances, "Number of thresholds " \
                                                  "must equal the amount of " \
                                                  "appliances"

        # CONSTANTS

        k_thresh = constant(thresholds)
        k_sigma = constant(sigma_c)

        # ARCHITECTURE

        # Input layer (batch, series_len, 1)
        inputs = Input(shape=(series_len, 1))

        # 1D Conv (batch, series_len, 16)
        # filters = 16, kernel_size = 4
        conv1 = Conv1D(16, 4, activation="relu", padding="same",
                       strides=1)(inputs)
        drop_conv1 = Dropout(dropout)(conv1)
        # 1D Conv (batch, series_len, 8)
        conv2 = Conv1D(8, 4, activation="relu", padding="same",
                       strides=1)(drop_conv1)
        drop_conv2 = Dropout(dropout)(conv2)

        # Bi-directional LSTMs (batch, series_len, 128)
        gru1 = Bidirectional(GRU(64, return_sequences=True, stateful=False),
                             merge_mode='concat')(drop_conv2)
        drop_gru1 = Dropout(dropout)(gru1)
        # Bi-directional LSTMs (batch, series_len, 256)
        gru2 = Bidirectional(GRU(128, return_sequences=True, stateful=False),
                             merge_mode='concat')(drop_gru1)
        drop_gru2 = Dropout(dropout)(gru2)

        # Dense layer (batch, series_len, 64)
        dense = Dense(64, activation='relu')(drop_gru2)
        drop_dense = Dropout(dropout)(dense)

        # Regression output
        # Fully Connected Layers (batch, series_len, num_appliances)
        regression = Dense(num_appliances, activation='linear',
                           name='regression')(drop_dense)

        # Classification output
        # Apply a sigmoid centered around the threshold value of each appliance
        subtract = Lambda(lambda x: (x - k_thresh) * k_sigma)(regression)
        # Fully Connected Layers (batch, series_len, num_appliances)
        classification = Activation(sigmoid, name='classification')(subtract)

        # COMPILE

        # Optimizer
        opt = Adam(learning_rate=learning_rate)

        # Compile the model
        model = Model(inputs=inputs,
                      outputs=[regression, classification])
        model.compile(loss={"regression": "mean_squared_error",
                            "classification": "binary_crossentropy"},
                      loss_weights={"regression": regression_weight,
                                    "classification": classification_weight},
                      optimizer=opt)

        self.model = model
