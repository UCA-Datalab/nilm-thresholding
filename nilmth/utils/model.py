import os

import numpy as np

from nilmth.data.dataloader import DataLoader
from nilmth.data.temporal import remove_directory, generate_temporal_data
from nilmth.model.conv import ConvModel
from nilmth.model.gru import GRUModel
from nilmth.utils.logging import logger
from nilmth.utils.scores import (
    score_dict_predictions,
    average_list_dict_scores,
)
from nilmth.utils.store_output import (
    generate_path_output_model,
    generate_path_output_model_params,
    store_scores,
    store_plots,
)


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


def train_many_models(path_data: str, path_output: str, config: dict):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_out_model = generate_path_output_model(path_output, config["name"])
    path_out_model_params = generate_path_output_model_params(path_out_model, config)

    # Load dataloader
    dataloader_train = DataLoader(path_data, subset="train", shuffle=True, **config)
    dataloader_validation = DataLoader(
        path_data, subset="validation", shuffle=True, **config
    )
    dataloader_test = DataLoader(path_data, subset="test", shuffle=False, **config)

    generate_temporal_data(dataloader_train, path="temp_train")
    generate_temporal_data(dataloader_validation, path="temp_valid")

    # Initialize lists
    list_dict_scores = [{}] * config["num_models"]
    time_elapsed = [0.0] * config["num_models"]
    dict_pred = {}

    for i in range(config["num_models"]):
        logger.debug(f"\nModel {i + 1}\n")

        # Initialize the model
        model = initialize_model(config)

        # Train
        time_elapsed[i] = model.train(
            dataloader_train,
            dataloader_validation,
        )

        # Store the model
        path_model = os.path.join(path_out_model_params, f"model_{i}.pth")
        model.save(path_model)

        dict_pred = model.predictions_to_dictionary(dataloader_test)
        list_dict_scores[i] = score_dict_predictions(dict_pred)

        store_scores(
            config,
            list_dict_scores[i],
            time_elapsed=time_elapsed[i],
            path_output=os.path.join(path_out_model_params, f"scores_{i}.txt"),
        )

    # List of dicts to unique dict scores
    dict_scores = average_list_dict_scores(list_dict_scores)

    # Store scores and plot
    store_scores(
        config,
        dict_scores,
        time_elapsed=np.mean(time_elapsed),
        path_output=os.path.join(path_out_model_params, "scores.txt"),
    )
    store_plots(dict_pred, path_output=path_out_model_params)

    remove_directory("temp_train")
    remove_directory("temp_valid")


def test_many_models(path_data: str, path_output: str, config: dict):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_out_model = generate_path_output_model(path_output, config["name"])
    path_out_model_params = generate_path_output_model_params(path_out_model, config)

    # Load dataloader
    dataloader_test = DataLoader(path_data, subset="test", shuffle=False, **config)

    # Initialize lists
    list_dict_scores = [{}] * config["num_models"]

    for i in range(config["num_models"]):
        logger.debug(f"\nModel {i + 1}\n")

        model = initialize_model(config)

        # Load the model
        path_model = os.path.join(path_out_model_params, f"model_{i}.pth")
        model.load(path_model)

        dict_pred = model.predictions_to_dictionary(dataloader_test)
        list_dict_scores[i] = score_dict_predictions(dict_pred)

        store_scores(
            config,
            list_dict_scores[i],
            path_output=os.path.join(path_out_model_params, f"scores_{i}.txt"),
        )

        del model

    # List of dicts to unique dict scores
    dict_scores = average_list_dict_scores(list_dict_scores)

    # Store scores and plot
    store_scores(
        config,
        dict_scores,
        path_output=os.path.join(path_out_model_params, "scores.txt"),
    )
