import os
import shutil

import numpy as np
import pandas as pd

from nilm_thresholding.data.loader import DataLoader
from nilm_thresholding.model.conv import ConvModel
from nilm_thresholding.model.gru import GRUModel
from nilm_thresholding.results.store_output import (
    generate_path_output,
    generate_folder_name,
    store_scores,
    list_scores,
    store_plots,
)
from nilm_thresholding.utils.format_list import merge_dict_list
from nilm_thresholding.utils.logging import logger


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


def remove_directory(path: str):
    """Removes a folder"""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def generate_temporal_data(loader: DataLoader, path: str = "data_temp"):
    """Stores data ready to be used by the model, with the statuses already computed"""
    # Create a new directory of temporal data
    remove_directory(path)
    os.mkdir(path)
    # Initialize the file number and the list of files
    file_num = 0
    files = loader.dataset.files.copy()
    # Iterate through the whole dataloader
    for data, target_power, target_status in iter(loader):
        data = data.cpu().detach().numpy()
        target_power = target_power.cpu().detach().numpy()
        target_status = target_status.cpu().detach().numpy()
        # Add the border for appliance power and status
        npad = ((0, 0), (loader.dataset.border, loader.dataset.border), (0, 0))
        target_power = np.pad(
            target_power, pad_width=npad, mode="constant", constant_values=0
        )
        target_status = np.pad(
            target_status, pad_width=npad, mode="constant", constant_values=0
        )
        # Stack all arrays
        mat = np.concatenate(
            [np.expand_dims(data, axis=2), target_power, target_status], axis=2
        )
        # Store each series in a different csv
        for m in mat:
            df = pd.DataFrame(
                m,
                columns=["aggregate"]
                + loader.dataset.appliances
                + loader.dataset.status,
            )
            path_file = os.path.join(path, f"{file_num:04}.csv")
            df.to_csv(path_file)
            # Add the file to the file list
            files[file_num] = path_file
            file_num += 1
    # Update the file list to match the temporal file list
    loader.dataset.files = files


def train_many_models(path_data, path_output, config):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_output = generate_path_output(path_output, config["name"])
    path_output_folder = generate_folder_name(path_output, config)

    # Load dataloader
    dataloader_train = DataLoader(path_data, subset="train", shuffle=True, **config)
    dataloader_validation = DataLoader(
        path_data, subset="validation", shuffle=True, **config
    )
    dataloader_test = DataLoader(path_data, subset="test", shuffle=False, **config)

    generate_temporal_data(dataloader_train, path="temp_train")
    generate_temporal_data(dataloader_validation, path="temp_valid")

    # Training

    act_scores = [0] * config["num_models"]
    pow_scores = [0] * config["num_models"]
    time_elapsed = [0.0] * config["num_models"]

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
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.save(path_model)

        act_scores[i], pow_scores[i] = model.score(dataloader_test)

        # Store individual scores
        act_dict = merge_dict_list(act_scores[i])
        pow_dict = merge_dict_list(pow_scores[i])

        scores = {"classification": act_dict, "regression": pow_dict}

        filename = f"scores_{i}.txt"

        store_scores(
            path_output_folder,
            config,
            scores,
            time_elapsed[i],
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
    store_scores(
        path_output_folder,
        config,
        scores,
        np.mean(time_elapsed),
    )

    remove_directory("temp_train")
    remove_directory("temp_valid")


def test_many_models(path_data, path_output, config):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """
    # Set output path
    path_output = generate_path_output(path_output, config["name"])
    path_output_folder = generate_folder_name(path_output, config)

    # Load dataloader
    dataloader_test = DataLoader(path_data, subset="test", shuffle=False, **config)

    # Training

    act_scores = []
    pow_scores = []
    time_elapsed = 0

    for i in range(config["num_models"]):
        logger.debug(f"\nModel {i + 1}\n")

        model = initialize_model(config)

        # Load the model
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.load(path_model)

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
            time_elapsed,
            filename=filename,
        )

        del model

    # List scores

    scores = list_scores(
        config["appliances"],
        act_scores,
        pow_scores,
        config["num_models"],
    )

    # Store scores and plot
    store_scores(path_output_folder, config, scores, time_elapsed)

    store_plots(
        path_output_folder,
        config,
        model,
        dataloader_test,
    )
