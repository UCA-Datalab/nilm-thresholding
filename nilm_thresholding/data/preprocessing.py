import os

import numpy as np
import pandas as pd

from nilm_thresholding.data.threshold import Threshold
from nilm_thresholding.utils.format_list import to_list


class PreprocessWrapper:
    dataset: str = "wrapper"

    def __init__(self, config: dict):
        # Read parameters from config files
        self.appliances = config["appliances"]
        self.buildings = to_list(config["buildings"][self.dataset])
        self.dates = config["dates"][self.dataset]
        self.period = config["period"]
        self.size = {
            "train": config["train_size"],
            "validation": config["valid_size"],
            "test": 1 - config["train_size"] - config["valid_size"],
        }
        self.input_len = config["input_len"]
        self.border = config["border"]
        self.max_power = config["max_power"]

        # Set the parameters according to given threshold method
        self.threshold = Threshold(self.appliances, **config.get("threshold", {}))

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
        ser = meters.drop("aggregate", axis=1).values
        status = self.threshold.get_status(ser)
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
            meters = self._get_status(meters)
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
