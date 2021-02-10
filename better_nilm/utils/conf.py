import toml
import warnings
from pathlib import Path
from typing import Union


class Conf(dict):
    """Sub-class of dict that overrides __getitem__ to allow for keys not in
    the original dict, defaulting to None.
    """

    def __init__(self, *args, **kwargs):
        """Update dict with all keys from dict"""
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        """Get key from dict. If not present, return None and raise warning
        Parameters
        ----------
        key : Hashable
            key to get from original dict
        Returns
        -------
            original value in the dict or None if not present
        """
        if key not in self:
            warnings.warn(f"Key '{key}' not in conf. Defaulting to None")
            val = None
        else:
            val = dict.__getitem__(self, key)
        return val


def load_conf(path: Union[str, Path], key: str = None) -> Conf:
    """Load TOML config as dict-like
    Parameters
    ----------
    path : str
        Path to TOML config file
    key : str, optional
        Section of the conf file to load
    Returns
    -------
    Conf
        Config dictionary
    """
    config = toml.load(path)
    return Conf(config) if key is None else Conf(config[key])


def load_conf_data(path: Union[str, Path]) -> Conf:
    """Load TOML data params as dict-like
    Parameters
    ----------
    path : str
        Path to TOML config file
    Returns
    -------
    Conf
        Config dictionary
    """
    config = load_conf(path, "data")
    buildings = sorted(
        set(
            config["building_train"]
            + config["building_valid"]
            + config["building_test"]
        )
    )
    config.update({"buildings": buildings})
    # Dates keys to string
    config_dates = config["dates"]
    config_dates_int = {}
    for k, v in config_dates.items():
        config_dates_int.update({int(k): v})
    config.update(config_dates_int)
    # If min_on of min_off are "None", change them to None
    config_threshold = config["threshold"]
    for k in ["min_on", "min_off"]:
        if config_threshold[k] == "None":
            config_threshold.update({k: None})
    config.update({"threshold": config_threshold})
    return config


def load_conf_model(path: Union[str, Path]) -> Conf:
    """Load TOML model params as dict-like
    Parameters
    ----------
    path : str
        Path to TOML config file
    Returns
    -------
    Conf
        Config dictionary
    """
    config = load_conf(path)
    config_model = config["model"]
    # Update model params
    model_params = config_model["params"]
    reg_w = 1 - model_params["classification_w"]
    num_app = len(config["data"]["appliances"])
    model_params.update({"out_channels": num_app, "regression_w": reg_w})
    # Update model config
    config_model.update({"params": model_params})
    return config_model


def load_conf_full(path: Union[str, Path]) -> Conf:
    """Load TOML dictionary, fully processed
    Parameters
    ----------
    path : str
        Path to TOML config file
    Returns
    -------
    Conf
        Config dictionary
    """
    config = load_conf(path)
    config_data = load_conf_data(path)
    config_model = load_conf_model(path)
    config.update({"data": config_data, "model": config_model})
    return config
