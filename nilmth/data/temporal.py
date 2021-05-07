import os
import shutil

import numpy as np
import pandas as pd

from nilmth.data.loader import DataLoader


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
