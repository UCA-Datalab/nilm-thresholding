import warnings
from pathlib import Path
from typing import Union

import toml


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
        return self._format_values(val)

    def _format_values(self, d: dict):
        try:
            for k, v in d.items():
                if type(v) is dict:
                    self._format_values(v)
                elif v == "None":
                    d.update({k: None})
        except AttributeError:
            pass
        return d


class ConfigError(Exception):
    """Exception when config file is not opened properly"""

    pass


def load_config(path: Union[str, Path], key: str = None) -> Conf:
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
    try:
        config = toml.load(path)
    except FileNotFoundError:
        raise ConfigError(f"Config toml file not found in path: {path}")
    try:
        return Conf(config) if key is None else Conf(config[key])
    except KeyError:
        raise ConfigError(f"Key missing in config file: {key}")


def store_config(path: Union[str, Path], data: dict):
    with open(path, "w") as toml_file:
        toml.dump(data, toml_file)
