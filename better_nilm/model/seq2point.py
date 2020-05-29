from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout

from keras.backend import constant
from keras.layers import Lambda
from keras.layers import Activation
from keras.activations import sigmoid

from keras.optimizers import Adam
    

def create_seq2point_model(series_len, num_appliances, thresholds,
                     regression_weight=1, classification_weight=1,
                     learning_rate=0.001, sigma_c=50):
    """
    Creates a Seq2Point model.
    Based on Krystalakos model:
    https://www.researchgate.net/publication/326238920_Sliding_Window_Approach_for_Online_Energy_Disaggregation_Using_Artificial_Neural_Networks
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

    Returns
    -------
    model : keras.models.Model

    """
    assert len(thresholds) == num_appliances, "Number of thresholds must " \
                                              "equal the amount of appliances"
        
    # CONSTANTS
    
    k_thresh = constant(thresholds)
    k_sigma = constant(sigma_c)

    # ARCHITECTURE

    # Input layer (batch, series_len, 1)
    inputs = Input(shape=(series_len, 1))

    # 1D Conv (batch, series_len, 30)
    # filters = 30, kernel_size = 10
    conv1 = Conv1D(30, 10, activation="relu", padding="same", strides=1)(
        inputs)
    drop1 = Dropout(0.5)(conv1)
    # 1D Conv (batch, series_len, 30)
    conv2 = Conv1D(30, 8, activation="relu", padding="same", strides=1)(drop1)
    drop2 = Dropout(0.5)(conv2)
    # 1D Conv (batch, series_len, 40)
    conv3 = Conv1D(40, 6, activation="relu", padding="same", strides=1)(drop2)
    drop3 = Dropout(0.5)(conv3)
    # 1D Conv (batch, series_len, 50)
    conv4 = Conv1D(50, 5, activation="relu", padding="same", strides=1)(drop3)
    drop4 = Dropout(0.5)(conv4)

    # Dense layer (batch, series_len, 1024)
    dense = Dense(1024, activation='relu')(drop4)
    drop_dense = Dropout(0.5)(dense)

    # Regression output
    # Fully Connected Layers (batch, series_len, num_appliances)
    regression = Dense(num_appliances, activation='linear',
                       name='regression')(drop_dense)

    # Classification output
    # Apply a sigmoid centered around the threshold value of each appliance
    subtract = Lambda(lambda x: (x - k_thresh) * k_sigma)(regression)
    # Fully Connected Layers (batch, series_len, num_appliances)
    classification = Activation(sigmoid, name='classification')(subtract)

    # TRAINING

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

    return model
