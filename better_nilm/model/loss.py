from keras import backend as K


def regression_loss():
    def rmse_loss(y_true, y_pred):
        rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
        return rmse
    return rmse_loss


def classification_loss():
    def rmse_loss(y_true, y_pred):
        rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
        return rmse

    return rmse_loss
