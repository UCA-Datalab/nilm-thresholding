import os
import random

import numpy as np
import pandas as pd
import torch.utils.data as data
from torch.utils.data import DataLoader

from nilm_thresholding.data.thresholding import get_status_by_duration


class DataSet(data.Dataset):
    files: list = list()
    epochs: int = 0
    appliances: list = list()
    status: list = list()

    def __init__(self, path_data: str, config: dict, subset: str = "train"):
        self.power_scale = config.get("power_scale", 2000)
        self.border = config.get("border", 15)
        self.length = config.get("input_len", 510)
        self.threshold = config.get("threshold", {})
        self._list_files(path_data, config, subset)
        self._get_parameters_from_file()

        # Thresholding parameters
        self.thresholds = np.zeros((len(self.appliances), 1))
        self.min_on = np.ones((len(self.appliances)))
        self.min_off = np.ones((len(self.appliances)))
        self.means = np.ones((len(self.appliances), 2))

    @staticmethod
    def _open_file(path_file: str) -> pd.DataFrame:
        """Opens a csv as a pandas.DataFrame"""
        df = pd.read_csv(path_file, index_col=0)
        return df

    def _list_files(self, path_data: str, config: dict, subset: str):
        """List the files pertaining to given subset"""
        # Initialize empty file list
        files = []
        # Loop through the datasets and buildings as sorted in config
        for dataset, buildings in config[subset]["buildings"].items():
            for building in buildings:
                path_building = os.path.join(path_data, f"{dataset}_{building}")
                files_of_building = sorted(
                    [
                        os.path.join(path_building, file)
                        for file in os.listdir(path_building)
                    ]
                )
                val_idx = int(len(files_of_building) * config["train_size"])
                test_idx = val_idx + int(len(files_of_building) * config["valid_size"])
                # Shuffle if requested
                if config.get("random", False):
                    random.seed(config.get("random_seed", 0))
                    random.shuffle(files_of_building)
                # Pick subset to choose from
                if subset == "train":
                    files += files_of_building[:val_idx]
                elif subset == "validation":
                    files += files_of_building[val_idx:test_idx]
                elif subset == "test":
                    files += files_of_building[test_idx:]
        # Update the class parameters
        self.files = files
        self.epochs = len(files)
        print(f"{self.epochs} data points found for {subset}")

    def _get_parameters_from_file(self):
        """Updates class parameters from sample csv file"""
        df = self._open_file(self.files[0])
        appliances = [t for t in df.columns if not t.endswith("_status")]
        appliances.remove("aggregate")
        self.appliances = appliances
        self.status = [t + "_status" for t in appliances]
        self.length = df.shape[0]
        self._idx_start = self.border
        self._idx_end = self.length - self.border

    def _compute_status(self, arr_apps: np.array):
        status = get_status_by_duration(
            arr_apps, self.thresholds, self.min_off, self.min_on
        )
        status = status.reshape(status.shape[0], len(self.appliances))
        return status

    def __getitem__(self, index):
        path_file = self.files[index]
        df = self._open_file(path_file)
        x = df["aggregate"].values / self.power_scale
        y = (
            df[self.appliances].iloc[self._idx_start : self._idx_end].values
            / self.power_scale
        )
        s = self._compute_status(y)
        #s = df[self.status].iloc[self._idx_start : self._idx_end].values
        return x, y, s

    def __len__(self):
        return self.epochs


def return_dataloader(
    path_data: str,
    config: dict,
    subset: str = "train",
    shuffle: bool = True,
):
    dataset = DataSet(path_data, config, subset=subset)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config["batch_size"], shuffle=shuffle
    )
    return dataloader
