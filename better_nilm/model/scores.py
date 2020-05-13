import numpy as np

from sklearn.metrics import mean_squared_error


def rmse_score(pred, real):
    assert pred.shape == real.shape, "Both predicted and real arrays must " \
                                     "have the same shape."

    num_appliances = pred.shape[2]
    rmse = np.zeros(num_appliances)

    for idx in range(num_appliances):
        p = pred[:, :, idx].copy()
        p = p.flatten()
        r = real[:, :, idx].copy()
        r = r.flatten()
        rmse[idx] = mean_squared_error(p, r)

    return rmse


def _assert_shape(y_pred, y_real, appliances):
    if not (y_pred.shape == y_real.shape).all():
        raise ValueError("Array shape mismatch.\n"
                         f"y_pred shape: {y_pred.shape}\n"
                         f"y_real_shape: {y_real.shape}")

    if y_pred.shape[2] != len(appliances):
        raise ValueError("Number of appliances mismatch.\n"
                         f"Appliances in y_pred array: {y_pred.shape[2]}\n"
                         f"Appliances in appliances list: {len(appliances)}")


def regression_score_dict(y_pred, y_real, appliances):
    _assert_shape(y_pred, y_real, appliances)

    if np.mean(y_real) <= 1:
        print("Warning!\nThe predicted values appear to be normalized.\n"
              "It is recommended to use the de-normalized values\n"
              "when computing the regression errors")

    # Initialize dict
    scores = {}

    for idx, app in enumerate(appliances):
        app_pred = y_pred[:, :, idx].copy()
        app_real = y_real[:, :, idx].copy()
        # RMSE score
        rmse = rmse_score(app_pred, app_real)

        scores[app] = {"rmse": rmse}

    return scores
