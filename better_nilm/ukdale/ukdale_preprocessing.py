import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from better_nilm.format_utils import to_list
from better_nilm.ukdale.ukdale_data import load_ukdale_series


class Power(data.Dataset):
    def __init__(self, meter=None, appliance=None, status=None,
                 length=256, border=680, max_power=1., train=False):
        self.length = length
        self.border = border
        self.max_power = max_power
        self.train = train

        self.meter = meter.copy() / self.max_power
        self.appliance = appliance.copy() / self.max_power
        self.status = status.copy()

        self.epochs = (len(self.meter) - 2 * self.border) // self.length

    def __getitem__(self, index):
        i = index * self.length + self.border
        if self.train:
            i = np.random.randint(self.border,
                                  len(self.meter) - self.length - self.border)

        x = self.meter.iloc[
            i - self.border:i + self.length + self.border].values.astype(
            'float32')
        y = self.appliance.iloc[i:i + self.length].values.astype('float32')
        s = self.status.iloc[i:i + self.length].values.astype('float32')
        x -= x.mean()

        return x, y, s

    def __len__(self):
        return self.epochs


def _train_valid_test(ds_meter, ds_appliance, ds_status, num_buildings,
                      train_size=0.8, valid_size=0.1,
                      seq_len=512, border=16, max_power=10000.):
    ds_len = [len(ds_meter[i]) for i in range(num_buildings)]

    ds_house_train = [Power(ds_meter[i][:int(train_size * ds_len[i])],
                            ds_appliance[i][:int(train_size * ds_len[i])],
                            ds_status[i][:int(train_size * ds_len[i])],
                            seq_len, border, max_power, True) for i in
                      range(num_buildings)]

    ds_house_valid = [
        Power(ds_meter[i][int(train_size * ds_len[i]):int(
            (train_size + valid_size) * ds_len[i])],
              ds_appliance[i][int(train_size * ds_len[i]):int(
                  (train_size + valid_size) * ds_len[i])],
              ds_status[i][int(train_size * ds_len[i]):int(
                  (train_size + valid_size) * ds_len[i])],
              seq_len, border, max_power, False) for i in range(num_buildings)]

    ds_house_test = [
        Power(ds_meter[i][int((train_size + valid_size) * ds_len[i]):],
              ds_appliance[i][int((train_size + valid_size) * ds_len[i]):],
              ds_status[i][int((train_size + valid_size) * ds_len[i]):],
              seq_len, border, max_power, False) for i in
        range(num_buildings)]
    return ds_house_train, ds_house_valid, ds_house_test


def _datastore_to_dataloader(ds_house, buildings, batch_size, shuffle):
    buildings = to_list(buildings)
    ds = []
    for building in buildings:
        ds += [ds_house[building]]
    ds = torch.utils.data.ConcatDataset(ds)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def datastores_to_dataloaders(ds_meter, ds_appliance, ds_status, num_buildings,
                              train_buildings, valid_buildings, test_buildings,
                              train_size=0.8, valid_size=0.1, batch_size=64,
                              seq_len=512, border=16, max_power=10000.):
    ds_house_train, \
    ds_house_valid, \
    ds_house_test = _train_valid_test(
        ds_meter, ds_appliance, ds_status, num_buildings,
        train_size=train_size, valid_size=valid_size,
        seq_len=seq_len, border=border, max_power=max_power)

    dl_train = _datastore_to_dataloader(ds_house_train, train_buildings,
                                        batch_size, True)
    dl_valid = _datastore_to_dataloader(ds_house_valid, valid_buildings,
                                        batch_size, False)
    dl_test = _datastore_to_dataloader(ds_house_test, test_buildings,
                                       batch_size, False)
    return dl_train, dl_valid, dl_test


def _buildings_to_idx(buildings, train_buildings, valid_buildings,
                      test_buildings):
    # Train, valid and test buildings must contain the index, not the ID of
    # the building. Change that
    if train_buildings is None:
        tr_buildings = [i for i in range(len(buildings))]
    else:
        tr_buildings = []

    if valid_buildings is None:
        vl_buildings = [i for i in range(len(buildings))]
    else:
        vl_buildings = []

    if test_buildings is None:
        ts_buildings = [i for i in range(len(buildings))]
    else:
        ts_buildings = []

    for idx, building in enumerate(buildings):
        if (train_buildings is not None) and (building in train_buildings):
            tr_buildings += [idx]
        if (valid_buildings is not None) and (building in valid_buildings):
            vl_buildings += [idx]
        if (test_buildings is not None) and (building in test_buildings):
            ts_buildings += idx

    assert len(tr_buildings) > 0, f"No ID in train_buildings matches the " \
                                  f"ones of buildings."
    assert len(vl_buildings) > 0, f"No ID in valid_buildings matches the " \
                                  f"ones of buildings."
    assert len(ts_buildings) > 0, f"No ID in test_buildings matches the " \
                                  f"ones of buildings."

    return tr_buildings, vl_buildings, ts_buildings


def load_dataloaders(path_h5, path_data, buildings, appliances,
                     train_buildings=None, valid_buildings=None,
                     test_buildings=None,
                     train_size=0.8, valid_size=0.1, batch_size=64,
                     seq_len=512, border=16, max_power=10000.):
    tr_buildings,\
    vl_buildings,\
    ts_buildings = _buildings_to_idx(buildings, train_buildings,
                                     valid_buildings, test_buildings)

    # Load the different datastores
    ds_meter, ds_appliance, ds_status = load_ukdale_series(path_h5, path_data,
                                                           buildings,
                                                           appliances)
    num_buildings = len(buildings)

    # Load the data loaders
    dl_train, \
    dl_valid, \
    dl_test = datastores_to_dataloaders(ds_meter, ds_appliance, ds_status,
                                        num_buildings, tr_buildings,
                                        vl_buildings, ts_buildings,
                                        train_size=train_size,
                                        valid_size=valid_size,
                                        batch_size=batch_size,
                                        seq_len=seq_len, border=border,
                                        max_power=max_power)
    return dl_train, dl_valid, dl_test
