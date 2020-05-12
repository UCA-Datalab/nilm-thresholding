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
