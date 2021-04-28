import os
import random

import numpy as np
import pandas as pd
import torch.utils.data as data

from nilm_thresholding.data.threshold import Threshold
from nilm_thresholding.utils.config import ConfigError
from nilm_thresholding.utils.logging import logger


class DataSet(data.Dataset):
    files: list = list()
    datapoints: int = 0
    appliances: list = list()
    status: list = list()

    def __init__(
        self,
        path_data: str,
        subset: str = "train",
        power_scale: int = 2000,
        input_len: int = 510,
        border: int = 15,
        buildings: dict = None,
        train_size: float = 0.8,
        valid_size: float = 0.1,
        random_split: bool = False,
        random_seed: int = 0,
        threshold: dict = None,
        **kwargs,
    ):
        self.subset = subset
        self.power_scale = power_scale
        self.border = border
        self.length = input_len
        self.buildings = {} if buildings is None else buildings[subset]
        self.train_size = train_size
        self.validation_size = valid_size
        self.random_split = random_split
        self.random_seed = random_seed
        self._list_files(path_data)
        self._get_parameters_from_file()

        # Set the parameters according to given threshold method
        param_thresh = {} if threshold is None else threshold
        self.threshold = Threshold(appliances=self.appliances, **param_thresh)

        logger.debug(
            f"Dataset received extra kwargs, not used:\n     {', '.join(kwargs.keys())}"
        )

    @staticmethod
    def _open_file(path_file: str) -> pd.DataFrame:
        """Opens a csv as a pandas.DataFrame"""
        df = pd.read_csv(path_file, index_col=0)
        return df

    def _list_files(self, path_data: str):
        """List the files pertaining to given subset"""
        # Initialize empty file list
        files = []
        # Loop through the datasets and buildings as sorted in config
        for dataset, buildings in self.buildings.items():
            for building in buildings:
                path_building = os.path.join(path_data, f"{dataset}_{building}")
                files_of_building = sorted(
                    [
                        os.path.join(path_building, file)
                        for file in os.listdir(path_building)
                    ]
                )
                val_idx = int(len(files_of_building) * self.train_size)
                test_idx = val_idx + int(len(files_of_building) * self.validation_size)
                # Shuffle if requested
                if self.random_split:
                    random.seed(self.random_seed)
                    random.shuffle(files_of_building)
                # Pick subset to choose from
                if self.subset == "train":
                    files += files_of_building[:val_idx]
                elif self.subset == "validation":
                    files += files_of_building[val_idx:test_idx]
                elif self.subset == "test":
                    files += files_of_building[test_idx:]
        # Update the class parameters
        self.files = files
        self.datapoints = len(files)
        logger.info(f"{self.datapoints} data points found for {self.subset}")

    def _get_parameters_from_file(self):
        """Updates class parameters from sample csv file"""
        df = self._open_file(self.files[0])
        appliances = [t for t in df.columns if not t.endswith("_status")]
        appliances.remove("aggregate")
        self.appliances = sorted(appliances)
        self.status = [app + "_status" for app in self.appliances]
        self.length = df.shape[0]
        self._idx_start = self.border
        self._idx_end = self.length - self.border

    def normalize_power(self, ser: np.array) -> np.array:
        return ser / self.power_scale

    def denormalize_power(self, ser: np.array) -> np.array:
        return ser * self.power_scale

    def _compute_status(self, ser: np.array) -> np.array:
        return self.threshold.get_status(ser)

    def __getitem__(self, index):
        path_file = self.files[index]
        df = self._open_file(path_file)
        x = df["aggregate"].values
        y = df[self.appliances].iloc[self._idx_start : self._idx_end].values
        try:
            s = df[self.status].iloc[self._idx_start : self._idx_end].values
        except KeyError:
            s = self._compute_status(y)
        return self.normalize_power(x), self.normalize_power(y), s

    def __len__(self):
        return self.datapoints


class DataLoader(data.DataLoader):
    dataset: DataSet = None

    def __init__(
        self,
        path_data: str,
        subset: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        path_threshold: str = "threshold.toml",
        **kwargs,
    ):
        self.subset = subset
        dataset = DataSet(path_data, subset=subset, **kwargs)
        super(DataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle
        )
        self.path_threshold = path_threshold
        self.compute_thresholds()

    def compute_thresholds(self):
        """Compute the thresholds of each appliance"""
        # First try to load the thresholds
        try:
            self.dataset.threshold.read_config(self.path_threshold)
            return
        # If not possible, compute them
        except ConfigError:
            if self.subset != "train":
                logger.error(
                    "Threshold values not found."
                    "Please compute them first with the train subset!"
                )
            logger.debug("Threshold values not found. Computing them...")
            # Loop through each appliance
            for app_idx, app in enumerate(self.dataset.appliances):
                # Initialize list
                ser = [0] * self.__len__()
                # Loop through the set and extract all the values
                for idx, (_, meters, _) in enumerate(self):
                    ser[idx] = meters[:, :, app_idx].flatten()
                # Concatenate all values and denormalize them
                ser = self.dataset.denormalize_power(np.concatenate(ser))
                self.dataset.threshold.update_appliance_threshold(ser, app)
            # Write the config file
            self.dataset.threshold.write_config(self.path_threshold)
