import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


def rmse_score(pred, real):
    assert pred.shape == real.shape, "Both predicted and real arrays must " \
                                     "have the same shape."

    num_appliances = pred.shape[2]
    mse = np.zeros(num_appliances)

    for idx in range(num_appliances):
        p = pred[:, :, idx].copy()
        p = p.flatten()
        r = real[:, :, idx].copy()
        r = r.flatten()
        mse[idx] = mean_squared_error(p, r)

    rmse = np.sqrt(mse)
    return rmse


def _assert_shape(y_pred, y_real, appliances):
    if not y_pred.shape == y_real.shape:
        raise ValueError("Array shape mismatch.\n"
                         f"y_pred shape: {y_pred.shape}\n"
                         f"y_real_shape: {y_real.shape}")

    if y_pred.shape[2] != len(appliances):
        raise ValueError("Number of appliances mismatch.\n"
                         f"Appliances in y_pred array: {y_pred.shape[2]}\n"
                         f"Appliances in appliances list: {len(appliances)}")


def regression_score_dict(y_pred, y_real, appliances):
    """
    Returns a dictionary with some regression scores, for each appliance.
        - RMSE, Root Mean Squared Error

    Parameters
    ----------
    y_pred : numpy.array
        shape = (num_series, series_len, num_appliances)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_appliances : Meters contained in the array.
    y_real : numpy.array
        shape = (num_series, series_len, num_appliances)
    appliances : list
        len = num_appliances
        Must be sorted following the order of both y_pred and y_real

    Returns
    -------
    scores : dict
        'appliance': {'metric': value}

    """
    _assert_shape(y_pred, y_real, appliances)

    if np.mean(y_real) <= 1:
        print("Warning!\nThe predicted values appear to be normalized.\n"
              "It is recommended to use the de-normalized values\n"
              "when computing the regression errors")

    # Initialize dict
    scores = {}

    # Compute RMSE for all appliances
    rmse = rmse_score(y_pred, y_real)

    for idx, app in enumerate(appliances):
        # RMSE
        app_rmse = round(rmse[idx], 2)

        scores[app] = {"rmse": app_rmse}

    return scores


def classification_scores_dict(y_pred, y_real, appliances, threshold=.5):
    """
    Returns a dictionary with some regression scores, for each appliance.
        -

    Parameters
    ----------
    y_pred : numpy.array
        shape = (num_series, series_len, num_appliances)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_appliances : Meters contained in the array.
    y_real : numpy.array
        shape = (num_series, series_len, num_appliances)
    appliances : list
        len = num_appliances
        Must be sorted following the order of both y_pred and y_real
    threshold : float, default=0.5
        Minimum value (form 0 to 1) at which we consider the appliance to be ON

    Returns
    -------
    scores : dict
        'appliance': {'metric': value}

    """

    _assert_shape(y_pred, y_real, appliances)

    if ((y_pred.max() > 1) or (y_real > 1)
            or (y_pred.min() < 0) or (y_real.min() < 0)):
        raise ValueError("Classification values must be between 0 and 1.")

    # Binarize the arrays
    bin_pred = y_pred.copy()
    bin_pred[y_pred < threshold] = 0
    bin_pred[y_pred >= threshold] = 1

    bin_real = y_real.copy()
    bin_real[y_real < threshold] = 0
    bin_real[y_real >= threshold] = 1

    # Initialize dict
    scores = {}

    for idx, app in enumerate(appliances):
        app_pred = bin_pred[:, :, idx]
        app_real = bin_real[:, :, idx]

        # F1-Score
        app_f1 = f1_score(app_real, app_pred)

        scores[app] = {"f1score": app_f1}

    return scores
