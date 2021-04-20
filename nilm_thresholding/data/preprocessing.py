import os

import numpy as np
import pandas as pd

from nilm_thresholding.data.thresholding import (
    get_status,
    get_status_by_duration,
    get_status_means,
    get_thresholds,
    get_threshold_params,
)
from nilm_thresholding.utils.format_list import to_list


class PreprocessWrapper:
    dataset: str = "wrapper"

    def __init__(self, config: dict):
        # Read parameters from config files
        self.appliances = config["appliances"]
        self.buildings = to_list(config[self.dataset]["buildings"])
        self.dates = config[self.dataset]["dates"]
        self.period = config["period"]
        self.size = {
            "train": config["train_size"],
            "validation": config["valid_size"],
            "test": 1 - config["train_size"] - config["valid_size"],
        }
        self.input_len = config["input_len"]
        self.border = config["border"]
        self.threshold_method = config["threshold"]["method"]
        self.threshold_std = config["threshold"]["std"]
        self.thresholds = config["threshold"]["list"]
        self.min_off = config["threshold"]["min_off"]
        self.min_on = config["threshold"]["min_on"]
        self.max_power = config["max_power"]

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
            self.means = get_status_means(arr_apps, status)
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
        # Loop through the buildings that are going to be stored
        for house in self.buildings:
            # Load the chosen meters of the building, compute their status
            meters = self.load_house_meters(house)
            # Check the number of data points
            step = self.input_len - self.border
            size = meters.shape[0] // step
            idx = 0
            # Store data points sequentially
            # Create the building folder inside each subset folder
            path_house = os.path.join(path_output, f"{self.dataset}_{house}")
            try:
                os.mkdir(path_house)
            except FileNotFoundError:
                os.mkdir(path_house)
            except FileExistsError:
                pass
            # Check the number of data points
            print(f"House {house}: {size} data points")
            for point in range(size):
                # Each data point is stored individually
                df_sub = meters.iloc[idx : (idx + self.input_len)]
                path_file = os.path.join(path_house, f"{point:04}.csv")
                # Sort columns by name
                df_sub = df_sub.reindex(sorted(df_sub.columns), axis=1)
                df_sub.to_csv(path_file)
                idx += step
