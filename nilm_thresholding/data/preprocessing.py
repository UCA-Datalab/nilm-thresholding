import logging
import os

import numpy as np
import pandas as pd

from nilm_thresholding.data.threshold import Threshold
from nilm_thresholding.utils.format_list import to_list

logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class PreprocessWrapper:
    dataset: str = "wrapper"

    def __init__(
        self,
        appliances: list = None,
        buildings: dict = None,
        dates: dict = None,
        period: str = "1min",
        train_size: float = 0.8,
        valid_size: float = 0.1,
        input_len: int = 510,
        border: int = 15,
        max_power: float = 10000,
        threshold: dict = None,
        **kwargs,
    ):
        # Read parameters from config files
        self.appliances = [] if appliances is None else to_list(appliances)
        self.buildings = [] if buildings is None else buildings[self.dataset]
        self.dates = [] if dates is None else dates[self.dataset]
        self.period = period
        self.size = {
            "train": train_size,
            "validation": valid_size,
            "test": 1 - train_size - valid_size,
        }
        self.input_len = input_len
        self.border = border
        self.max_power = max_power

        # Set the parameters according to given threshold method
        param_thresh = {} if threshold is None else threshold
        self.threshold = Threshold(appliances=self.appliances, **param_thresh)

        logging.debug(
            f"Received extra kwargs, not used:\n   {', '.join(kwargs.keys())}"
        )

    def load_house_meters(self, house: int) -> pd.DataFrame:
        """Placeholder function, this should load the household meters and status"""
        return pd.DataFrame()

    def compute_thresholds(self, buildings: list = None):
        """Compute the thresholds of each appliance, using the given list of buildings"""
        buildings = self.buildings if buildings is None else buildings
        for app in self.appliances:
            ser = [0] * len(buildings)
            idx = 0
            for house in buildings:
                # Load the chosen meters of the building, compute their status
                meters = self.load_house_meters(house)
                ser[idx] = meters[app].values
                idx += 1
            ser = np.concatenate(ser)
            self.threshold.update_appliance_threshold(ser, app)

    def _include_status(self, meters: pd.DataFrame):
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

    def store_preprocessed_data(self, path_output: str):
        """Stores preprocessed data in output folder"""
        self.compute_thresholds(buildings=[1])
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
