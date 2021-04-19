# Shut Future Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import time

import typer

from nilm_thresholding.data.loader import return_dataloader
from nilm_thresholding.model.model import initialize_model
from nilm_thresholding.results.store_output import (
    generate_path_output,
    generate_folder_name,
    list_scores,
    store_scores,
)
from nilm_thresholding.utils.config import load_config
from nilm_thresholding.utils.format_list import merge_dict_list


def train_many_models(path_data, path_output, config_data, config_model):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_output = generate_path_output(path_output, config_model["name"])
    path_output_folder = generate_folder_name(
        path_output,
        config_data["input_len"],
        config_data["period"],
        config_model["classification_w"],
        config_model["regression_w"],
        config_data["threshold"]["method"],
    )

    # Load dataloader
    dataloader_train = return_dataloader(
        path_data, config_data, config_model, subset="train", shuffle=True
    )
    dataloader_validation = return_dataloader(
        path_data, config_data, config_model, subset="validation", shuffle=False
    )
    dataloader_test = return_dataloader(
        path_data, config_data, config_model, subset="test", shuffle=False
    )

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0
    config_model["out_channels"] = len(config_data["appliances"])

    for i in range(config_model["num_models"]):
        print(f"\nModel {i + 1}\n")

        model = initialize_model(config_model)

        # Train
        time_start = time.time()
        model.train_with_dataloader(
            dataloader_train,
            dataloader_validation,
        )
        time_ellapsed += time.time() - time_start

        # Store the model
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.save(path_model)

        act_scr, pow_scr = model.get_scores(dataloader_test)

        act_scores += act_scr
        pow_scores += pow_scr

        # Store individual scores
        act_dict = merge_dict_list(act_scr)
        pow_dict = merge_dict_list(pow_scr)

        scores = {"classification": act_dict, "regression": pow_dict}

        filename = f"scores_{i}.txt"

        store_scores(
            path_output_folder,
            config,
            scores,
            time_ellapsed,
            filename=filename,
        )

    # List scores

    scores = list_scores(
        config["data"]["appliances"],
        act_scores,
        pow_scores,
        config["train"]["num_models"],
    )

    time_ellapsed /= config["train"]["num_models"]

    # Store scores and plot

    store_scores(
        path_output_folder,
        config,
        scores,
        time_ellapsed,
    )


def main(
    path_data: str = "data-prep",
    path_output: str = "outputs",
    path_config: str = "nilm_thresholding/config.toml",
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
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config_data = load_config(path_config, "data")
    config_model = load_config(path_config, "model")
    print("Done\n")

    # Run main results
    print(f"{config_model['name']}\n")

    train_many_models(path_data, path_output, config_data, config_model)


if __name__ == "__main__":
    typer.run(main)
