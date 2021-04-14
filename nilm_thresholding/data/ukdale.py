import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from pandas import Series
from pandas.io.pytables import HDFStore
from torch.utils.data import DataLoader

from nilm_thresholding.data.wrapper import DataloaderWrapper
from nilm_thresholding.model.preprocessing import (
    get_status,
    get_status_by_duration,
    get_status_means,
    get_thresholds,
)
from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.string import APPLIANCE_NAMES, homogenize_string


class Power(data.Dataset):
    def __init__(
        self,
        meter=None,
        appliance=None,
        status=None,
        length=512,
        border=16,
        power_scale=2000.0,
        train=False,
    ):
        self.length = length
        self.border = border
        self.power_scale = power_scale
        self.train = train

        self.meter = meter.copy() / self.power_scale
        self.appliance = appliance.copy() / self.power_scale
        self.status = status.copy()

        self.epochs = (len(self.meter) - 2 * self.border) // self.length

    def __getitem__(self, index):
        i = index * self.length + self.border
        if self.train:
            i = np.random.randint(
                self.border, len(self.meter) - self.length - self.border
            )

        x = self.meter.iloc[
            i - self.border : i + self.length + self.border
        ].values.astype("float32")
        y = self.appliance.iloc[i : i + self.length].values.astype("float32")
        s = self.status.iloc[i : i + self.length].values.astype("float32")
        x -= x.mean()

        return x, y, s

    def __len__(self):
        return self.epochs


class UkdaleDataloader(DataloaderWrapper):
    list_df_meter: list = []
    list_df_appliance: list = []
    list_df_status: list = []
    means: list = []
    dl_train: DataLoader = None
    dl_valid: DataLoader = None
    dl_test: DataLoader = None

    def __init__(self, path_h5: str, path_labels: str, config: dict):
        super(UkdaleDataloader, self).__init__(config)
        # Load the different datastores
        self.load_series(path_h5, path_labels)

        # Load the data loaders
        self.datastores_to_dataloaders()

    @staticmethod
    def load_datastore(path_h5: str) -> pd.HDFStore:
        """
        Loads the UKDALE h5 file as a datastore.

        Parameters
        ----------
        path_h5 : str
            Path to the original UKDALE h5 file

        Returns
        -------
        datastore : pandas.HDFStore
        """
        assert os.path.isfile(path_h5), (
            f"Input path does not lead to file:" f"\n{path_h5}"
        )
        assert path_h5.endswith(".h5"), (
            "Path must lead to a h5 file.\n" f"Input is {path_h5}"
        )
        datastore = pd.HDFStore(path_h5)
        return datastore

    def load_meter(
        self, datastore: pd.HDFStore, building: int, meter: int
    ) -> pd.Series:
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
        assert type(datastore) is HDFStore, (
            "datastore must be "
            "pandas.io.pytables.HDFStore\n"
            f"Input is {type(datastore)}"
        )
        key = "/building{}/elec/meter{}".format(building, meter)
        m = datastore[key]
        v = m.values.flatten()
        t = m.index
        s = pd.Series(v, index=t).clip(0.0, self.max_power)
        s[s < 10.0] = 0.0
        s = s.resample("1s").ffill(limit=300).fillna(0.0)
        s = s.resample(self.period).mean().tz_convert("UTC")
        return s

    def datastore_to_series(
        self, path_labels: str, datastore: pd.HDFStore, house: int, label: str
    ) -> pd.Series:
        """

        Parameters
        ----------
        path_labels : str
            Path to the directory that contains the csv of the meter labels.
        datastore : pandas.HDFStore
        house : int
            Building ID
        label : str
            Meter name

        Returns
        -------
        s : pandas.Series
        """
        # Load the meter labels
        assert os.path.isdir(
            path_labels
        ), f"Input path is not a directory:\n{path_labels}"
        filename = f"{path_labels}/house_%1d/labels.dat" % house
        assert os.path.isfile(filename), f"Path not found:\n{filename}"

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
                s = self.load_meter(datastore, house, i)

        if s is None:
            raise ValueError(
                f"Label {label} not found on house {house}\n"
                f"Valid labels are: {list(labels.values())}"
            )

        msg = (
            f"load_ukdale_meter() should output {Series}\n"
            f"Received {type(s)} instead"
        )
        assert type(s) is Series, msg

        s.index.name = "datetime"
        s.name = label

        return s

    def load_series(self, path_h5: str, path_labels: str):
        # Load datastore
        datastore = self.load_datastore(path_h5)
        assert type(datastore) is HDFStore, (
            f"load_ukdale_datastore() should output {HDFStore}\n"
            f"Received {type(datastore)} instead"
        )

        # Make a list of meters
        list_meters = self.appliances.copy()
        list_meters.append("aggregate")

        # Initialize list
        for house in self.buildings:
            meters = []
            for m in list_meters:
                series_meter = self.datastore_to_series(
                    path_labels, datastore, house, m
                )
                meters += [series_meter]

            meters = pd.concat(meters, axis=1)
            meters.fillna(method="pad", inplace=True)
            msg = (
                f"meters dataframe must have {len(list_meters)} columns\n"
                f"It currently has {meters.shape[1]} "
            )
            assert meters.shape[1] == len(list_meters), msg

            # Pick range of dates
            try:
                date_start = self.dates[house][0]
                date_start = pd.to_datetime(date_start).tz_localize("Etc/UCT")
                date_end = self.dates[house][1]
                date_end = pd.to_datetime(date_end).tz_localize("Etc/UCT")

                msg = (
                    f"Start date is {date_start}\nEnd date is {date_end}\n"
                    "End date must be after start date!"
                )
                assert date_end > date_start, msg

                meters = meters[date_start:date_end]

                msg = (
                    "meters dataframe was left empty after applying dates\n"
                    f"Start date is {date_start}\nEnd date is {date_end}"
                )
                assert meters.shape[0] > 0, msg
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

                msg = "Number of thresholds doesn't match number of appliances"
                assert len(self.thresholds) == len(self.appliances), msg

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

            self.list_df_meter.append(meter)
            self.list_df_appliance.append(appliances)
            self.list_df_status.append(status)

    def _train_valid_test(self):
        """
        Splits data store data into train, validation and tests.
        """
        df_len = [len(self.list_df_meter[i]) for i in range(self.num_buildings)]

        ds_train = [
            Power(
                self.list_df_meter[i][: int(self.train_size * df_len[i])],
                self.list_df_appliance[i][: int(self.train_size * df_len[i])],
                self.list_df_status[i][: int(self.train_size * df_len[i])],
                self.output_len,
                self.border,
                self.power_scale,
                train=True,
            )
            for i in range(self.num_buildings)
        ]

        ds_valid = [
            Power(
                self.list_df_meter[i][
                    int(self.train_size * df_len[i]) : int(
                        (self.train_size + self.valid_size) * df_len[i]
                    )
                ],
                self.list_df_appliance[i][
                    int(self.train_size * df_len[i]) : int(
                        (self.train_size + self.valid_size) * df_len[i]
                    )
                ],
                self.list_df_status[i][
                    int(self.train_size * df_len[i]) : int(
                        (self.train_size + self.valid_size) * df_len[i]
                    )
                ],
                self.output_len,
                self.border,
                self.power_scale,
                train=False,
            )
            for i in range(self.num_buildings)
        ]

        ds_test = [
            Power(
                self.list_df_meter[i][
                    int((self.train_size + self.valid_size) * df_len[i]) :
                ],
                self.list_df_appliance[i][
                    int((self.train_size + self.valid_size) * df_len[i]) :
                ],
                self.list_df_status[i][
                    int((self.train_size + self.valid_size) * df_len[i]) :
                ],
                self.output_len,
                self.border,
                self.power_scale,
                train=False,
            )
            for i in range(self.num_buildings)
        ]

        return ds_train, ds_valid, ds_test

    def _datastore_to_dataloader(self, list_ds, build_idx, shuffle):
        """
        Turns a datastore into a dataloader.
        """
        build_idx = to_list(build_idx)
        ds = []
        for idx in build_idx:
            ds += [list_ds[idx]]
        ds = torch.utils.data.ConcatDataset(ds)
        dl = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=shuffle)
        return dl

    def datastores_to_dataloaders(self):
        """
        Turns datastores into dataloaders.
        """
        ds_train, ds_valid, ds_test = self._train_valid_test()

        self.dl_train = self._datastore_to_dataloader(
            ds_train, self.build_idx_train, True
        )
        self.dl_valid = self._datastore_to_dataloader(
            ds_valid, self.build_idx_valid, False
        )
        self.dl_test = self._datastore_to_dataloader(
            ds_test, self.build_idx_test, False
        )
