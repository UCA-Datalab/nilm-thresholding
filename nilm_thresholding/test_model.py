# Shut Future Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import typer

from nilm_thresholding.model.model import initialize_model
from nilm_thresholding.results.plot_output import plot_scores_by_class_weight
from nilm_thresholding.results.store_output import (
    generate_path_output,
    generate_folder_name,
    list_scores,
    store_plots,
    store_scores,
    store_real_data_and_predictions,
)
from nilm_thresholding.utils.config import load_config
from nilm_thresholding.utils.format_list import merge_dict_list
from nilm_thresholding.data.loader import return_dataloader


def test_many_models(
    path_h5,
    path_data,
    path_output,
    config: dict,
    save_scores: bool = True,
    save_predictions: bool = True,
):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """

    # Set output path
    path_output = generate_path_output(path_output, config["name"])
    path_output_folder = generate_folder_name(
        path_output,
        config["output_len"],
        config["period"],
        config["classification_w"],
        config["regression_w"],
        config["threshold"]["method"],
    )

    # Load data

    dataloader = return_dataloader(path_data, config)

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0

    for i in range(config["num_models"]):
        print(f"\nModel {i + 1}\n")

        model = initialize_model(config)

        # Load model
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.load(path_model)

        act_scr, pow_scr = model.get_scores(
            model,
            dataloader.dl_test,
        )

        act_scores += act_scr
        pow_scores += pow_scr

        # Store individual scores
        act_dict = merge_dict_list(act_scr)
        pow_dict = merge_dict_list(pow_scr)

        scores = {"classification": act_dict, "regression": pow_dict}

        filename = f"scores_{i}.txt"

        if save_scores:
            store_scores(
                path_output_folder,
                config,
                scores,
                time_ellapsed,
                filename=filename,
            )

    # List scores

    scores = list_scores(
        config["appliances"],
        act_scores,
        pow_scores,
        config["num_models"],
    )

    # Store scores and plot
    if save_scores:
        store_scores(path_output_folder, config, scores, time_ellapsed)

    store_plots(
        path_output_folder,
        config,
        model,
        dataloader.dl_test,
        dataloader.means,
        dataloader.thresholds,
    )

    if save_predictions:
        store_real_data_and_predictions(
            path_output,
            config,
            model,
            dataloader.dl_test,
            dataloader.means,
            dataloader.thresholds,
        )


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
