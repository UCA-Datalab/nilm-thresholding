# Shut Future Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import typer

from nilm_thresholding.model.train import test_many_models
from nilm_thresholding.results.plot_output import plot_scores_by_class_weight
from nilm_thresholding.utils.config import load_config


def main(
    path_data: str = "data-prep",
    path_output: str = "outputs",
    path_config: str = "nilm_thresholding/config.toml",
    save_scores: bool = True,
    save_predictions: bool = True,
):
    """
    Trains several CONV models under the same conditions
    Stores scores and plots on results folder

    Parameters
    ----------
    path_data : str, optional
        Path to UK-DALE data
    path_output : str, optional
        Path to the results folder
    path_config : str, optional
        Path to the config toml file
    save_scores : bool, optional
        Store the model scores in txt files, by default True
    save_predictions : bool, optional
        Store the model predictions in txt files, by default True
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_config(path_config, "model")
    print("Done\n")

    assert os.path.isdir(path_data), "path_data must lead to folder:\n{}".format(
        path_data
    )
    path_h5 = path_data + ".h5"
    assert os.path.isfile(path_h5), "File not found:\n{}".format(path_h5)

    # Run main results
    print(f"{config['name']}\n")

    test_many_models(
        path_h5,
        path_data,
        path_output,
        config,
        save_scores=save_scores,
        save_predictions=save_predictions,
    )

    print("PLOT RESULTS!")
    plot_scores_by_class_weight(config, path_output)


if __name__ == "__main__":
    typer.run(main)
