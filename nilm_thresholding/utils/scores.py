import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def regression_scores_dict(
    dict_pred: dict, key_real: str = "power", key_pred: str = "power_pred"
):
    """
    Returns a dictionary with some regression scores, for each appliance.
        - MSE, Mean Square Error
        - RMSE, Root Mean Squared Error

    Parameters
    ----------
    dict_pred : dict
        Dictionary containing all the relevant data
        returned by model.predictions_to_dictionary
    key_real : str, optional
        Key of the real value, by default "power"
    key_pred : str, optional
        Key of the predicted value, by default "power_pred"

    Returns
    -------
    scores : dict
        'appliance': {'metric': value}

    """

    # Initialize dict
    scores = {}

    for app, values in dict_pred.items():
        # Skip aggregated power load
        if app == "aggregated":
            continue

        app_real = values[key_real]
        app_pred = values[key_pred]

        # MSE and RMSE
        app_mse = mean_squared_error(app_real, app_pred)
        app_rmse = np.sqrt(app_mse)

        # MAE
        app_mae = mean_absolute_error(app_real, app_pred)

        # SAE (Signal Aggregate Error)
        app_sae = (np.sum(app_pred) - np.sum(app_real)) / np.sum(app_real)

        # NDE (Normalized Disaggregation Error)
        app_nde = np.sqrt(
            np.divide(
                np.sum(np.power((app_real - app_pred), 2)),
                np.sum(np.power(app_pred, 2)),
            )
        )

        # Energy based precision and recall
        energy_precision = np.divide(
            np.sum(np.minimum(app_real, app_pred)), np.sum(app_pred)
        )
        energy_recall = np.divide(
            np.sum(np.minimum(app_real, app_pred)), np.sum(app_real)
        )

        scores[app] = {
            "mse": round(app_mse, 2),
            "rmse": round(app_rmse, 2),
            "mae": round(app_mae, 2),
            "sae": round(app_sae, 2),
            "nde": round(app_nde, 4),
            "energy_precision": round(energy_precision, 4),
            "energy_recall": round(energy_recall, 4),
        }

    return scores


def classification_scores_dict(
    dict_pred: dict, key_real: str = "status", key_pred: str = "status_pred"
):
    """
    Returns a dictionary with some regression scores, for each appliance.
        - Accuracy
        - F1-Score
        - Precision
        - Recall

    Parameters
    ----------
    dict_pred : dict
        Dictionary containing all the relevant data
        returned by model.predictions_to_dictionary
    key_real : str, optional
        Key of the real value, by default "status"
    key_pred : str, optional
        Key of the predicted value, by default "status_pred"

    Returns
    -------
    scores : dict
        'appliance': {'metric': value}

    """

    # Initialize dict
    scores = {}

    for app, values in dict_pred.items():
        # Skip aggregated power load
        if app == "aggregated":
            continue

        app_real = values[key_real].astype(int)
        app_pred = values[key_pred].astype(int)

        # Precision
        app_accuracy = accuracy_score(app_real, app_pred)

        # F1-Score
        app_f1 = f1_score(app_real, app_pred)

        # Precision
        app_precision = precision_score(app_real, app_pred)

        # Recall
        app_recall = recall_score(app_real, app_pred)

        scores[app] = {
            "accuracy": round(app_accuracy, 4),
            "f1": round(app_f1, 4),
            "precision": round(app_precision, 4),
            "recall": round(app_recall, 4),
        }

    return scores
