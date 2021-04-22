import os
import time

from nilm_thresholding.data.loader import DataLoader
from nilm_thresholding.model.conv import ConvModel
from nilm_thresholding.model.gru import GRUModel
from nilm_thresholding.results.store_output import (
    generate_path_output,
    generate_folder_name,
    store_scores,
    list_scores,
    store_plots,
    store_real_data_and_predictions,
)
from nilm_thresholding.utils.format_list import merge_dict_list


def initialize_model(config: dict):
    """Initialize a model given a config dictionary"""
    model_name = config["name"].lower()
    if model_name.startswith("conv"):
        model = ConvModel(**config)
    elif model_name.startswith("gru"):
        model = GRUModel(**config)
    else:
        raise ValueError(f"'{model_name}' not valid. Try using 'conv' or 'gru'")
    return model


def train_many_models(path_data, path_output, config):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_output = generate_path_output(path_output, config["name"])
    path_output_folder = generate_folder_name(
        path_output,
        config["input_len"],
        config["period"],
        config["classification_w"],
        config["regression_w"],
        config["threshold"]["method"],
    )

    # Load dataloader
    dataloader_train = DataLoader(path_data, subset="train", shuffle=True, **config)
    dataloader_validation = DataLoader(
        path_data, subset="validation", shuffle=True, **config
    )
    dataloader_test = DataLoader(path_data, subset="test", shuffle=False, **config)

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0

    for i in range(config["num_models"]):
        print(f"\nModel {i + 1}\n")

        model = initialize_model(config)

        # Train
        time_start = time.time()
        model.train(
            dataloader_train,
            dataloader_validation,
        )
        time_ellapsed += time.time() - time_start

        # Store the model
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.save(path_model)

        act_scr, pow_scr = model.score(dataloader_test)

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
        config["appliances"],
        act_scores,
        pow_scores,
        config["num_models"],
    )

    time_ellapsed /= config["num_models"]

    # Store scores and plot

    store_scores(
        path_output_folder,
        config,
        scores,
        time_ellapsed,
    )


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

        act_scr, pow_scr = model.score(
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
