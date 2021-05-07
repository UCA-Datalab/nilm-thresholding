import os

import typer

from nilm_thresholding.data.ukdale import UkdalePreprocess
from nilm_thresholding.utils.config import load_config
from nilm_thresholding.utils.logging import logger


def preprocess_ukdale(path_data: str, path_output: str, config: dict):
    """Preprocess and store UK-DALE data"""
    path_ukdale = os.path.join(path_data, "ukdale")

    path_h5 = path_ukdale + ".h5"

    prep = UkdalePreprocess(path_h5, path_ukdale, **config)
    prep.store_preprocessed_data(path_output)


def main(
    path_data: str = "data",
    path_output: str = "data-prep",
    path_config: str = "nilm_thresholding/config.toml",
):
    """Preprocess the raw data according to the configuration stated in config.toml
    Store the preprocessed data in other folder. This preprocessed data will then be
    used by the model.

    Parameters
    ----------
    path_data : str, optional
        Path to raw data, by default "data"
    path_output : str, optional
        Path to the results folder, by default "data-prep"
    path_config : str, optional
        Path to the config toml file, by defaul "nilm_thresholding/config.toml"
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_config(path_config, "model")
    print("Done\n")
    # Preprocess UK-DALE data
    try:
        preprocess_ukdale(path_data, path_output, config)
    except FileNotFoundError:
        logger.warning(f"UK-DALE not found in path: {path_data}")


if __name__ == "__main__":
    typer.run(main)
