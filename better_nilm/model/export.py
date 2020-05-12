import os

from keras.models import model_from_json


def store_model_json(model, path_model, path_weights=None):
    """
    Serializes a model into a json file.
    Also serializes its weights as a h5 file.
    Parameters
    ----------
    model : keras.models.Sequential
    path_model : str
        Path to where the json is created, including the filename and json
        termination.
    path_weights : str, default=None
        Path to where the h5 is created, including the filename and h5
        termination. If None is provided, weights are stored in the same
        route as the model, using the same name.

    """
    if not path_model.endswith(".json"):
        raise ValueError("path_model must end in a json file. Current "
                         f"route:\n{path_model}")

    if path_weights is None:
        path_weights = path_model.rsplit(".")[0]
        path_weights = path_weights + ".h5"
    elif not path_weights.endswith(".h5"):
        raise ValueError("path_weights must end in a h5 file. Current "
                         f"route:\n{path_weights}")
    # serialize model to JSON
    model_json = model.to_json()
    with open(path_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path_weights)
    print(f"Saved model to disk. Path:\n{path_model}")


def load_model_json(path_model, path_weights=None):
    """

    Parameters
    ----------
    path_model : str
        Path to where the serialized model is stored, in json format.
    path_weights : str, default=None
        Path to where the model weights are stored, in h5 format.
        If None is provided, assumes the h5 file is located in the same route
        as the model and with the same name.

    Returns
    -------
    model : keras.models.Sequential

    """
    if not path_model.endswith(".json"):
        raise ValueError("path_model must end in a json file. Current "
                         f"route:\n{path_model}")

    if path_weights is None:
        path_weights = path_model.rsplit(".")[0]
        path_weights = path_weights + ".h5"
    elif not path_weights.endswith(".h5"):
        raise ValueError("path_weights must end in a h5 file. Current "
                         f"route:\n{path_weights}")

    if not os.path.isfile(path_model):
        raise FileNotFoundError(f"path_model does not lead to an existing "
                                f"file:\n{path_model}")

    if not os.path.isfile(path_weights):
        raise FileNotFoundError(f"path_weights does not lead to an existing "
                                f"file:\n{path_weights}")

    # load json and create model
    json_file = open(path_model, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(path_weights)
    print("Loaded model from disk")
    return model
