import os

import numpy as np
import pandas as pd
from pandas import Series
from pandas.io.pytables import HDFStore

from nilm_thresholding.data.wrapper import DataloaderWrapper
from nilm_thresholding.model.preprocessing import (
    get_status,
    get_status_by_duration,
    get_status_means,
    get_thresholds,
)
from nilm_thresholding.utils.string import APPLIANCE_NAMES, homogenize_string


class UkdaleDataloader(DataloaderWrapper):
    datastore: HDFStore = None
    means: list = []

    def __init__(self, path_h5: str, path_labels: str, config: dict):
        super(UkdaleDataloader, self).__init__(config)
        # Load the different datastores
        self._load_series(path_h5, path_labels)

        # Load the data loaders
        self._list_series_to_dataloaders()

    def _load_datastore(self, path_h5: str):
        """
        Loads the UKDALE h5 file as a datastore.

        Parameters
        ----------
        path_h5 : str
            Path to the original UKDALE h5 file

        """
        assert os.path.isfile(path_h5), (
            f"Input path does not lead to file:" f"\n{path_h5}"
        )
        assert path_h5.endswith(".h5"), (
            "Path must lead to a h5 file.\n" f"Input is {path_h5}"
        )
        self.datastore = pd.HDFStore(path_h5)

    def _load_meter(self, building: int, meter: int) -> pd.Series:
        """
        Loads an UKDALE meter from the datastore, and resamples it to given period.

        Parameters
        ----------
        datastore : pandas.HDFStore
        building : int
            Building ID.
        meter : int
            Meter ID.

        Returns
        -------
        s : pandas.Series
        """
        key = "/building{}/elec/meter{}".format(building, meter)
        m = self.datastore[key]
        v = m.values.flatten()
        t = m.index
        s = pd.Series(v, index=t).clip(0.0, self.max_power)
        s[s < 10.0] = 0.0
        s = s.resample("1s").ffill(limit=300).fillna(0.0)
        s = s.resample(self.period).mean().tz_convert("UTC")
        return s

    def _datastore_to_series(
        self, path_labels: str, house: int, label: str
    ) -> pd.Series:
        """Extracts a specific label (appliance) from a house, given the datastore

        Parameters
        ----------
        house : int
            Building ID
        label : str
            Meter name

        Returns
        -------
        s : pandas.Series
        """
        # Load the meter labels
        filename = f"{path_labels}/house_%1d/labels.dat" % house

        labels = pd.read_csv(
            filename, delimiter=" ", header=None, index_col=0
        ).to_dict()[1]

        # Homogenize input label
        label = homogenize_string(label)
        label = APPLIANCE_NAMES.get(label, label)

        # Series placeholder
        s = None

        # Iterate through all the existing labels, searching for the input label
        for i in labels:
            lab = homogenize_string(labels[i])
            lab = APPLIANCE_NAMES.get(lab, lab)
            # When we find the input label, we load the meter records
            if lab == label:
                print(i, labels[i])
                s = self._load_meter(house, i)

        if s is None:
            raise ValueError(
                f"Label {label} not found on house {house}\n"
                f"Valid labels are: {list(labels.values())}"
            )

        assert type(s) is Series, (
            f"load_ukdale_meter() should output {Series}\n"
            f"Received {type(s)} instead"
        )

        s.index.name = "datetime"
        s.name = label

        return s

    def _load_series(self, path_h5: str, path_labels: str):
        # Load datastore
        self._load_datastore(path_h5)

        # Make a list of meters
        list_meters = self.appliances.copy()
        list_meters.append("aggregate")

        # Initialize list
        for house in self.buildings:
            meters = []
            for m in list_meters:
                series_meter = self._datastore_to_series(path_labels, house, m)
                meters += [series_meter]

            meters = pd.concat(meters, axis=1)
            meters.fillna(method="pad", inplace=True)
            assert meters.shape[1] == len(list_meters), (
                f"meters dataframe must have {len(list_meters)} columns\n"
                f"It currently has {meters.shape[1]} "
            )

            # Pick range of dates
            try:
                date_start = self.dates[house][0]
                date_start = pd.to_datetime(date_start).tz_localize("Etc/UCT")
                date_end = self.dates[house][1]
                date_end = pd.to_datetime(date_end).tz_localize("Etc/UCT")

                assert date_end > date_start, (
                    f"Start date is {date_start}\nEnd date is {date_end}\n"
                    "End date must be after start date!"
                )

                meters = meters[date_start:date_end]

                assert meters.shape[0] > 0, (
                    "meters dataframe was left empty after applying dates\n"
                    f"Start date is {date_start}\nEnd date is {date_end}"
                )
            except KeyError:
                raise KeyError(f"House not found: {house}, of type {type(house)}")

            meter = meters["aggregate"]
            appliances = meters.drop("aggregate", axis=1)

            arr_apps = np.expand_dims(appliances.values, axis=1)
            if (
                (self.thresholds is None)
                or (self.min_on is None)
                or (self.min_off is None)
            ):
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
            status = pd.DataFrame(
                status, columns=self.appliances, index=appliances.index
            )

            assert (
                status.shape[0] == appliances.shape[0]
            ), "Number of records between appliance status and load doesn't match"

            self.list_series_meter.append(meter)
            self.list_series_appliance.append(appliances)
            self.list_series_status.append(status)
