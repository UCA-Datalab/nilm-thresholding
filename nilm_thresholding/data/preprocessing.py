import logging
import os

import pandas as pd

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
        input_len: int = 510,
        border: int = 15,
        max_power: float = 10000,
        **kwargs,
    ):
        """Basic preprocessing class, with no dataset associated

        Parameters
        ----------
        appliances : list, optional
            List of appliances, by default empty
        buildings : dict, optional
            Datasets and buildings to extract from each {dataset: [buildings]},
            by default empty
        dates : dict, optional
            Date period per building {dataset: {building: [start, end]}},
            by default empty
        period : str, optional
            Time series period, by default '1min'
        train_size : float, optional
            Train subset proportion, by default 0.8
        input_len : int, optional
            Length of input time series, by default 510
        border : int, optional
            Border of the series that is lost to output, by default 15
        max_power : float, optional
            Maximum power (watts), higher values are reduced to this, by default 10000
        kwargs
            Additional arguments, not taken into account
        """
        # Read parameters from config files
        self.appliances = [] if appliances is None else sorted(to_list(appliances))
        self.buildings = [] if buildings is None else buildings[self.dataset]
        self.dates = [] if dates is None else dates[self.dataset]
        self.period = period
        self.train_size = train_size
        self.input_len = input_len
        self.border = border
        self.step = self.input_len - self.border
        self.max_power = max_power

        logging.debug(
            f"Preprocessing received extra kwargs, not used:\n"
            f"    {', '.join(kwargs.keys())}\n"
        )

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
            size = meters.shape[0] // self.step
            idx = 0
            # Store data points sequentially
            # Create the building folder inside each subset folder
            path_house = os.path.join(path_output, f"{self.dataset}_{house}")
            try:
                os.mkdir(path_house)
            except FileNotFoundError:
                os.mkdir(path_output)
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
                idx += self.step
