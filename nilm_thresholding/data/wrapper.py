import os

import numpy as np
import pandas as pd

from nilm_thresholding.data.preprocessing import (
    get_status,
    get_status_by_duration,
    get_status_means,
    get_thresholds,
)
from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.threshold import get_threshold_params


class PreprocessWrapper:
    name: str = "wrapper"

    def __init__(self, config: dict):
        # Read parameters from config files
        self.dates = config["data"]["dates"]
        self.period = config["data"]["period"]
        self.size = {
            "train": config["train"]["train_size"],
            "validation": config["train"]["valid_size"],
            "test": 1 - config["train"]["train_size"] - config["train"]["valid_size"],
        }
        self.batch_size = config["train"]["batch_size"]
        self.output_len = config["train"]["model"]["output_len"]
        self.border = config["train"]["model"]["border"]
        self.input_len = self.output_len + 2 * self.border
        self.power_scale = config["data"]["power_scale"]
        self.max_power = config["data"]["max_power"]
        self.return_means = config["train"]["return_means"]
        self.threshold_method = config["data"]["threshold"]["method"]
        self.threshold_std = config["data"]["threshold"]["std"]
        self.thresholds = config["data"]["threshold"]["list"]
        self.min_off = config["data"]["threshold"]["min_off"]
        self.min_on = config["data"]["threshold"]["min_on"]
        self.buildings = to_list(config["data"]["buildings"])
        self.appliances = config["data"]["appliances"]

        # Set the parameters according to given threshold method
        if self.threshold_method != "custom":
            (
                self.thresholds,
                self.min_off,
                self.min_on,
                self.threshold_std,
            ) = get_threshold_params(self.threshold_method, self.appliances)

    def _get_status(self, meters: pd.DataFrame):
        """Includes the status columns for each device

        Parameters
        ----------
        meters : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Same dataframe, with status columns

        """
        arr_apps = np.expand_dims(meters.drop("aggregate", axis=1).values, axis=1)
        if (self.thresholds is None) or (self.min_on is None) or (self.min_off is None):
            self.thresholds, self.means = get_thresholds(
                arr_apps, use_std=self.threshold_std, return_mean=True
            )

            assert len(self.thresholds) == len(
                self.appliances
            ), "Number of thresholds doesn't match number of appliances"

            status = get_status(arr_apps, self.thresholds)
        else:
            status = get_status_by_duration(
                arr_apps, self.thresholds, self.min_off, self.min_on
            )
            self.means = get_status_means(arr_apps, status)
        status = status.reshape(status.shape[0], len(self.appliances))
        status = pd.DataFrame(status, columns=self.appliances, index=meters.index)
        meters = meters.merge(
            status,
            how="inner",
            on=None,
            left_on=None,
            right_on=None,
            left_index=True,
            right_index=True,
            sort=False,
            suffixes=("", "_status"),
            copy=True,
            indicator=False,
            validate=None,
        )
        return meters

    def load_house_meters(self, house: int) -> pd.DataFrame:
        """Placeholder function, this should load the household meters and status"""
        return pd.DataFrame()

    def store_preprocessed_data(self, path_output: str):
        """Stores preprocessed data in output folder"""
        # Create a folder for the three subsets
        for subset in ["train", "validation", "test"]:
            path_subset = os.path.join(path_output, subset)
            try:
                os.mkdir(path_subset)
            except FileNotFoundError:
                os.mkdir(path_output)
                os.mkdir(path_subset)
            except FileExistsError:
                pass
        # Loop through the buildings that are going to be stored
        for house in self.buildings:
            # Load the chosen meters of the building, compute their status
            meters = self.load_house_meters(house)
            meters = self._get_status(meters)
            # Check the number of data points
            size = meters.shape[0] // self.input_len
            idx = 0
            # Loop through the subsets and store data points sequentially
            for subset in ["train", "validation", "test"]:
                # Create the building folder inside each subset folder
                path_subset = os.path.join(path_output, subset)
                path_house = os.path.join(path_subset, f"{self.name}_{house}")
                try:
                    os.mkdir(path_house)
                except FileNotFoundError:
                    os.mkdir(path_house)
                except FileExistsError:
                    pass
                # Check the number of data points in that subset
                size_sub = int(self.size[subset] * size)
                for point in range(size_sub):
                    # Each data point is stored individually
                    df_sub = meters.iloc[idx : (idx + self.input_len)]
                    path_file = os.path.join(path_house, f"{point:04}.csv")
                    df_sub.to_csv(path_file)
                    idx += self.input_len
