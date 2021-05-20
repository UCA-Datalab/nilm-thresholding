from typing import Union

import numpy as np
import torch.utils.data as data

from nilmth.data.dataset import DataSet
from nilmth.data.threshold import Threshold
from nilmth.utils.config import ConfigError
from nilmth.utils.logging import logger


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

    @property
    def threshold(self) -> Threshold:
        return self.dataset.threshold

    @property
    def appliances(self) -> list:
        return self.dataset.appliances

    @property
    def status(self) -> list:
        return self.dataset.status

    @property
    def files(self) -> list:
        return self.dataset.files

    @property
    def border(self) -> int:
        return self.dataset.border

    def __repr__(self):
        """This message is returned any time the object is called"""
        return f"Dataloader > {self.dataset}"

    def get_appliance_series(
        self, app: Union[str, int], target: str = "power"
    ) -> np.array:
        """Returns the full series of power of an appliance

        Parameters
        ----------
        app : str
            Appliance label
        target : str, optional
            Target value (power or status), by default "power"

        Returns
        -------
        numpy.array

        """
        if type(app) == str:
            try:
                app_idx = self.appliances.index(app)
            except ValueError:
                raise ValueError(f"Appliance not found: {app}")
        else:
            app_idx = app
        # Initialize list
        ser = [0] * self.__len__()
        # Loop through the set and extract all the values
        if target.lower() == "power":
            for idx, (_, meters, _) in enumerate(self):
                ser[idx] = meters[:, :, app_idx].flatten()
        elif target.lower() == "status":
            for idx, (_, _, meters) in enumerate(self):
                ser[idx] = meters[:, :, app_idx].flatten()
        else:
            raise ValueError(f"Target not available: {target}")
        return np.concatenate(ser)

    def compute_thresholds(self):
        """Compute the thresholds of each appliance"""
        # First try to load the thresholds
        try:
            self.threshold.read_config(self.path_threshold)
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
            for app_idx, app in enumerate(self.appliances):
                ser = self.get_appliance_series(app_idx)
                # Concatenate all values and update the threshold
                self.threshold.update_appliance_threshold(ser, app)
            # Write the config file
            self.threshold.write_config(self.path_threshold)

    def next(self):
        """Returns the next data batch"""
        return next(self.__iter__())
