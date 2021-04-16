import os

import typer

from nilm_thresholding.data.ukdale import UkdalePreprocess
from nilm_thresholding.utils.conf import load_conf_full, update_config


def main(
    path_data: str = "data/ukdale",
    path_output: str = "data-prep",
    path_config: str = "nilm_thresholding/config.toml",
):
    """
    Trains several CONV models under the same conditions
    Stores scores and plots on results folder

    Parameters
    ----------
    path_data : str, optional
        Path to data
    path_output : str, optional
        Path to the results folder
    path_config : str, optional
        Path to the config toml file
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_conf_full(path_config)
    config = update_config(config)
    print("Done\n")

    assert os.path.isdir(path_data), "path_data must lead to folder:\n{}".format(
        path_data
    )
    path_h5 = path_data + ".h5"
    assert os.path.isfile(path_h5), "File not found:\n{}".format(path_h5)

    dataloader = UkdalePreprocess(path_h5, path_data, config)
    dataloader.store_preprocessed_data(path_output)


if __name__ == "__main__":
    typer.run(main)
