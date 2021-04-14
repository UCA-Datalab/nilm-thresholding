import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.threshold import get_threshold_params


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


class DataloaderWrapper:
    build_idx_train: list = list()
    build_idx_valid: list = list()
    build_idx_test: list = list()
    list_series_meter: list = []
    list_series_appliance: list = []
    list_series_status: list = []
    dl_train: DataLoader = None
    dl_valid: DataLoader = None
    dl_test: DataLoader = None

    def __init__(self, config: dict):
        # Read parameters from config files
        self.build_id_train = config["data"]["building_train"]
        self.build_id_valid = config["data"]["building_valid"]
        self.build_id_test = config["data"]["building_test"]
        self.dates = config["data"]["dates"]
        self.period = config["data"]["period"]
        self.train_size = config["train"]["train_size"]
        self.valid_size = config["train"]["valid_size"]
        self.batch_size = config["train"]["batch_size"]
        self.output_len = config["train"]["model"]["output_len"]
        self.border = config["train"]["model"]["border"]
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

        self.num_buildings = len(self.buildings)
        self._buildings_to_idx()

        # Set the parameters according to given threshold method
        if self.threshold_method != "custom":
            (
                self.thresholds,
                self.min_off,
                self.min_on,
                self.threshold_std,
            ) = get_threshold_params(self.threshold_method, self.appliances)

    def _buildings_to_idx(self):
        """
        Takes the list of buildings ID and changes them to their corresponding
        index.
        """

        # Train, valid and tests buildings must contain the index, not the ID of
        # the building. Change that
        if self.build_id_train is None:
            self.build_idx_train = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_train = []

        if self.build_id_valid is None:
            self.build_idx_valid = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_valid = []

        if self.build_id_test is None:
            self.build_idx_test = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_test = []

        for idx, building in enumerate(self.buildings):
            if (self.build_id_train is not None) and (building in self.build_id_train):
                self.build_idx_train += [idx]
            if (self.build_id_valid is not None) and (building in self.build_id_valid):
                self.build_idx_valid += [idx]
            if (self.build_id_test is not None) and (building in self.build_id_test):
                self.build_idx_test += [idx]

        assert (
            len(self.build_idx_train) > 0
        ), f"No ID in build_id_train matches the ones of buildings."
        assert (
            len(self.build_idx_valid) > 0
        ), f"No ID in build_id_valid matches the ones of buildings."
        assert (
            len(self.build_idx_test) > 0
        ), f"No ID in build_id_test matches the ones of buildings."

    def _train_valid_test(self):
        """
        Splits lists of pandas.Series data into train, validation and tests.
        """
        df_len = [len(self.list_series_meter[i]) for i in range(self.num_buildings)]

        ds_train = [
            Power(
                self.list_series_meter[i][: int(self.train_size * df_len[i])],
                self.list_series_appliance[i][: int(self.train_size * df_len[i])],
                self.list_series_status[i][: int(self.train_size * df_len[i])],
                self.output_len,
                self.border,
                self.power_scale,
                train=True,
            )
            for i in range(self.num_buildings)
        ]

        ds_valid = [
            Power(
                self.list_series_meter[i][
                    int(self.train_size * df_len[i]) : int(
                        (self.train_size + self.valid_size) * df_len[i]
                    )
                ],
                self.list_series_appliance[i][
                    int(self.train_size * df_len[i]) : int(
                        (self.train_size + self.valid_size) * df_len[i]
                    )
                ],
                self.list_series_status[i][
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
                self.list_series_meter[i][
                    int((self.train_size + self.valid_size) * df_len[i]) :
                ],
                self.list_series_appliance[i][
                    int((self.train_size + self.valid_size) * df_len[i]) :
                ],
                self.list_series_status[i][
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

    def _list_series_to_dataloader(self, list_series, build_idx, shuffle):
        """
        Turns a list of pandas.Series into a dataloader.
        """
        ds = []
        for idx in build_idx:
            ds += [list_series[idx]]
        ds = torch.utils.data.ConcatDataset(ds)
        dl = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=shuffle)
        return dl

    def _list_series_to_dataloaders(self):
        """
        Turns list of pandas.Series into dataloaders.
        """
        ds_train, ds_valid, ds_test = self._train_valid_test()

        self.dl_train = self._list_series_to_dataloader(
            ds_train, self.build_idx_train, True
        )
        self.dl_valid = self._list_series_to_dataloader(
            ds_valid, self.build_idx_valid, False
        )
        self.dl_test = self._list_series_to_dataloader(
            ds_test, self.build_idx_test, False
        )
