import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data as data

from pandas import Series
from pandas.io.pytables import HDFStore
from torch.utils.data import DataLoader

from better_nilm.format_utils import to_list
from better_nilm.str_utils import homogenize_string
from better_nilm.model.preprocessing import get_thresholds
from better_nilm.model.preprocessing import get_status
from better_nilm.model.preprocessing import get_status_by_duration

APPLIANCE_NAMES = {
    "freezer": "fridge",
    "fridgefreezer": "fridge",
    "washerdryer": "washingmachine"
}


def load_ukdale_datastore(path_h5):
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
    assert os.path.isfile(path_h5), f"Input path does not lead to file:" \
                                    f"\n{path_h5}"
    assert path_h5.endswith('.h5'), "Path must lead to a h5 file.\n" \
                                    f"Input is {path_h5}"
    datastore = pd.HDFStore(path_h5)
    return datastore


def load_ukdale_meter(datastore, building=1, meter=1, period='1min',
                      max_power=10000.):
    """
    Loads an UKDALE meter from the datastore, and resamples it to given period.
    
    Parameters
    ----------
    datastore : pandas.HDFStore
    building : int, default=1
        Building ID.
    meter : int, default=1
        Meter ID.
    period : str, default='1min'
        Sample period. Time between records.
    max_power : float, default=10000.
        Maximum load. Any value higher than this is decreased to match this 
        value.

    Returns
    -------
    s : pandas.Series
    """
    assert type(datastore) is HDFStore, "datastore must be " \
                                        "pandas.io.pytables.HDFStore\n" \
                                        f"Input is {type(datastore)}"
    key = '/building{}/elec/meter{}'.format(building, meter)
    m = datastore[key]
    v = m.values.flatten()
    t = m.index
    s = pd.Series(v, index=t).clip(0., max_power)
    s[s < 10.] = 0.
    s = s.resample('1s').ffill(limit=300).fillna(0.)
    s = s.resample(period).mean().tz_convert('UTC')
    return s


def ukdale_datastore_to_series(path_labels, datastore, house, label,
                               period='1min', max_power=10000.,
                               verbose=True):
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
    period : str, default='1min'
    max_power : float, default=10000.
    verbose : bool, default=True

    Returns
    -------
    s : pandas.Series
    """
    # Load the meter labels
    assert os.path.isdir(path_labels), "Input path is not a directory:" \
                                       f"\n{path_labels}"
    filename = f"{path_labels}/house_%1d/labels.dat" % house
    assert os.path.isfile(filename), f"Path not found:\n{filename}"

    if verbose:
        print(filename)

    labels = pd.read_csv(filename, delimiter=' ',
                         header=None, index_col=0).to_dict()[1]

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
            s = load_ukdale_meter(datastore, house, i, period, max_power)

    if s is None:
        raise ValueError(f"Label {label} not found on house {house}\n"
                         f"Valid labels are: {list(labels.values())}")

    msg = f"load_ukdale_meter() should output {Series}\n" \
          f"Received {type(s)} instead"
    assert type(s) is Series, msg

    s.index.name = 'datetime'
    s.name = label

    return s


def load_ukdale_series(path_h5, path_labels, buildings, list_appliances,
                       dates=None, period='1min', max_power=10000.,
                       thresholds=None, min_off=None, min_on=None,
                       threshold_std=True, return_kmeans=False):
    """
    
    Parameters
    ----------
    path_h5 : str
        Path to the original UKDALE h5 file
    path_labels : str
        Path to the directory that contains the csv of the meter labels.
    buildings : list
        List of buildings IDs. List of integers.
    list_appliances : list
        List of appliances labels. List of strings.
    dates : dict, default=None
        {building_id : (date_start, date_end)}
        Both dates are strings with format: 'YY-MM-DD'
    period : str, default='1min'
        Record frequency
    max_power : float, default=10000.
        Maximum load value
    thresholds : numpy.array, default=None
        shape = (num_meters,)
        Thresholds per appliance, in watts. If not provided, thresholds are 
        computed using k-means
    min_off : numpy.array, default=None
        shape = (num_meters,)
        Number of records that an appliance must be below the threshold to 
        be considered turned OFF. If not provided, thresholds are 
        computed using k-means
    min_on : numpy.array, default=None
        shape = (num_meters,)
        Number of records that an appliance must be above the threshold to 
        be considered turned ON. If not provided, thresholds are 
        computed using k-means
    threshold_std : bool, default=True
        If the threshold is computed by k-means, use the standard deviation 
        of each cluster to move the threshold
    return_kmeans : bool, default=False
        If True, also return the computed thresholds and means

    Returns
    -------
    list_df_meter : list
        List of dataframes.
    list_df_appliance : list
        List of dataframes.
    list_df_status : list
        List of dataframes.
    """
    # Load datastore
    datastore = load_ukdale_datastore(path_h5)
    msg = f"load_ukdale_datastore() should output {HDFStore}\n" \
          f"Received {type(datastore)} instead"
    assert type(datastore) is HDFStore, msg

    # Ensure both parameters are lists
    buildings = to_list(buildings)
    list_appliances = to_list(list_appliances)
    # Make a list of meters
    list_meters = list_appliances.copy()
    list_meters.append('aggregate')

    # Initialize list
    list_df_meter = []
    list_df_appliance = []
    list_df_status = []

    for house in buildings:
        meters = []
        for m in list_meters:
            series_meter = ukdale_datastore_to_series(path_labels, datastore,
                                                      house, m,
                                                      max_power=max_power,
                                                      period=period)
            meters += [series_meter]

        meters = pd.concat(meters, axis=1)
        meters.fillna(method='pad', inplace=True)
        msg = f"meters dataframe must have {len(list_meters)} columns\n" \
              f"It currently has {meters.shape[1]} "
        assert meters.shape[1] == len(list_meters), msg

        # Pick range of dates
        if (type(dates) == dict) and (house in dates.keys()):
            date_start = dates[house][0]
            date_start = pd.to_datetime(date_start).tz_localize('Etc/UCT')
            date_end = dates[house][1]
            date_end = pd.to_datetime(date_end).tz_localize('Etc/UCT')

            msg = f"Start date is {date_start}\nEnd date is {date_end}\n" \
                  "End date must be after start date!"
            assert date_end > date_start, msg

            meters = meters[date_start:date_end]

            msg = "meters dataframe was left empty after applying dates\n" \
                  f"Start date is {date_start}\nEnd date is {date_end}"
            assert meters.shape[0] > 0, msg

        meter = meters['aggregate']
        appliances = meters.drop('aggregate', axis=1)

        arr_apps = np.expand_dims(appliances.values, axis=1)
        if (thresholds is None) or (min_on is None) or (min_off is None):
            thresholds, means = get_thresholds(arr_apps, use_std=threshold_std,
                                               return_mean=True)

            msg = "Number of thresholds doesn't match number of appliances"
            assert len(thresholds) == len(list_appliances), msg

            status = get_status(arr_apps, thresholds)
        else:
            means = None
            status = get_status_by_duration(arr_apps, thresholds,
                                            min_off, min_on)
        status = status.reshape(status.shape[0], len(list_appliances))
        status = pd.DataFrame(status, columns=list_appliances,
                              index=appliances.index)

        msg = "Number of records between appliance status and load doesn't " \
              "match"
        assert status.shape[0] == appliances.shape[0], msg

        list_df_meter.append(meter)
        list_df_appliance.append(appliances)
        list_df_status.append(status)

    return_params = (list_df_meter, list_df_appliance, list_df_status)
    if (return_kmeans) and (means is not None):
        return_params += ((thresholds, means), )

    return return_params


class Power(data.Dataset):
    def __init__(self, meter=None, appliance=None, status=None,
                 length=512, border=16, power_scale=2000., train=False):
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


def _train_valid_test(list_df_meter, list_df_appliance, list_df_status,
                      num_buildings,
                      train_size=0.8, valid_size=0.1,
                      seq_len=512, border=16, power_scale=2000.):
    """
    Splits data store data into train, validation and test.
    Parameters
    ----------
    list_df_meter
    list_df_appliance
    list_df_status
    num_buildings
    train_size
    valid_size
    seq_len
    border
    power_scale

    Returns
    -------
    ds_train
    ds_valid
    ds_test
    """
    df_len = [len(list_df_meter[i]) for i in range(num_buildings)]

    ds_train = [Power(list_df_meter[i][:int(train_size * df_len[i])],
                      list_df_appliance[i][:int(train_size * df_len[i])],
                      list_df_status[i][:int(train_size * df_len[i])],
                      seq_len, border, power_scale, train=True) for i in
                range(num_buildings)]

    ds_valid = [
        Power(list_df_meter[i][int(train_size * df_len[i]):int(
            (train_size + valid_size) * df_len[i])],
              list_df_appliance[i][int(train_size * df_len[i]):int(
                  (train_size + valid_size) * df_len[i])],
              list_df_status[i][int(train_size * df_len[i]):int(
                  (train_size + valid_size) * df_len[i])],
              seq_len, border, power_scale, train=False) for i in
        range(num_buildings)]

    ds_test = [
        Power(list_df_meter[i][int((train_size + valid_size) * df_len[i]):],
              list_df_appliance[i][
              int((train_size + valid_size) * df_len[i]):],
              list_df_status[i][int((train_size + valid_size) * df_len[i]):],
              seq_len, border, power_scale, train=False) for i in
        range(num_buildings)]

    return ds_train, ds_valid, ds_test


def _datastore_to_dataloader(list_ds, build_idx, batch_size, shuffle):
    """
    Turns a datastore into a dataloader.
    
    Parameters
    ----------
    list_ds
    build_idx
    batch_size
    shuffle

    Returns
    -------
    dl
    """
    build_idx = to_list(build_idx)
    ds = []
    for idx in build_idx:
        ds += [list_ds[idx]]
    ds = torch.utils.data.ConcatDataset(ds)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def datastores_to_dataloaders(list_df_meter, list_df_appliance, list_df_status,
                              num_buildings,
                              build_id_train, build_id_valid, build_id_test,
                              train_size=0.8, valid_size=0.1, batch_size=64,
                              seq_len=512, border=16, power_scale=2000.):
    """
    Turns datastores into dataloaders.
    
    Parameters
    ----------
    list_df_meter
    list_df_appliance
    list_df_status
    num_buildings
    build_id_train
    build_id_valid
    build_id_test
    train_size
    valid_size
    batch_size
    seq_len
    border
    power_scale

    Returns
    -------
    dl_train
    dl_valid
    dl_test

    """
    ds_train, \
    ds_valid, \
    ds_test = _train_valid_test(
        list_df_meter, list_df_appliance, list_df_status, num_buildings,
        train_size=train_size, valid_size=valid_size,
        seq_len=seq_len, border=border, power_scale=power_scale)

    dl_train = _datastore_to_dataloader(ds_train, build_id_train,
                                        batch_size, True)
    dl_valid = _datastore_to_dataloader(ds_valid, build_id_valid,
                                        batch_size, False)
    dl_test = _datastore_to_dataloader(ds_test, build_id_test,
                                       batch_size, False)
    return dl_train, dl_valid, dl_test


def _buildings_to_idx(buildings, build_id_train, build_id_valid,
                      build_id_test):
    """
    Takes the list of buildings ID and changes them to their corresponding 
    index.
    
    Parameters
    ----------
    buildings
    build_id_train
    build_id_valid
    build_id_test

    Returns
    -------
    build_idx_train
    build_idx_valid
    build_idx_test

    """
    # Train, valid and test buildings must contain the index, not the ID of
    # the building. Change that
    if build_id_train is None:
        build_idx_train = [i for i in range(len(buildings))]
    else:
        build_idx_train = []

    if build_id_valid is None:
        build_idx_valid = [i for i in range(len(buildings))]
    else:
        build_idx_valid = []

    if build_id_test is None:
        build_idx_test = [i for i in range(len(buildings))]
    else:
        build_idx_test = []

    for idx, building in enumerate(buildings):
        if (build_id_train is not None) and (building in build_id_train):
            build_idx_train += [idx]
        if (build_id_valid is not None) and (building in build_id_valid):
            build_idx_valid += [idx]
        if (build_id_test is not None) and (building in build_id_test):
            build_idx_test += [idx]

    assert len(build_idx_train) > 0, f"No ID in build_id_train matches the " \
                                     f"ones of buildings."
    assert len(build_idx_valid) > 0, f"No ID in build_id_valid matches the " \
                                     f"ones of buildings."
    assert len(build_idx_test) > 0, f"No ID in build_id_test matches the " \
                                    f"ones of buildings."

    return build_idx_train, build_idx_valid, build_idx_test


def load_dataloaders(path_h5, path_labels, buildings, appliances,
                     dates=None, period='1min', max_power=10000.,
                     build_id_train=None, build_id_valid=None,
                     build_id_test=None,
                     train_size=0.8, valid_size=0.1, batch_size=64,
                     seq_len=480, border=16, power_scale=2000.,
                     thresholds=None, min_off=None, min_on=None,
                     threshold_std=True, return_kmeans=False):
    """
    Load the UKDALE dataloaders from the raw data.
    
    Parameters
    ----------
    path_h5 : str
        Path to the original UKDALE h5 file
    path_labels : str
        Path to the directory that contains the csv of the meter labels.
    buildings : list
        List of buildings IDs. List of integers.
    appliances : list
        List of appliances labels. List of strings.
    dates : dict, default=None
        {building_id : (date_start, date_end)}
        Both dates are strings with format: 'YY-MM-DD'
    build_id_train : list
        Buildings to use when training the model
    build_id_valid : list
        Buildings to use when validating the model
    build_id_test : list
        Buildings to use when testing the model
    train_size : float
        Train proportion. Must be between 0 and 1
    valid_size : float
        Train proportion. Must be between 0 and 1
    batch_size : int
    seq_len : int
        Length of the sequence to output
        len(output) = seq_len
    border : int
        Borders of the input sequence, that will be lost in the output.
        len(input) = seq_len + 2 * border
    period : float, default='1min'
        Record frequency
    max_power : float, default=10000.
        Maximum load power, in watts. Any value higher than max_power will be
        changed to this one
    power_scale : float, default=2000.
        Value by which we divide the power load values, to normalize them
    thresholds : numpy.array, default=None
        shape = (num_meters,)
        Thresholds per appliance, in watts. If not provided, thresholds are 
        computed using k-means
    min_off : numpy.array, default=None
        shape = (num_meters,)
        Number of records that an appliance must be below the threshold to 
        be considered turned OFF. If not provided, thresholds are 
        computed using k-means
    min_on : numpy.array, default=None
        shape = (num_meters,)
        Number of records that an appliance must be above the threshold to 
        be considered turned ON. If not provided, thresholds are 
        computed using k-means
    threshold_std : bool, default=True
        If the threshold is computed by k-means, use the standard deviation 
        of each cluster to move the threshold
    return_kmeans : bool, default=False
        If True, return as last argument the computed thresholds

    Returns
    -------
    dl_train
    dl_valid
    dl_test

    """
    build_idx_train, \
    build_idx_valid, \
    build_idx_test = _buildings_to_idx(buildings, build_id_train,
                                       build_id_valid, build_id_test)

    # Load the different datastores
    params = load_ukdale_series(path_h5, path_labels, buildings, appliances,
                                dates=dates, period=period,
                                max_power=max_power, thresholds=thresholds,
                                min_off=min_off, min_on=min_on,
                                threshold_std=threshold_std,
                                return_kmeans=return_kmeans)
    if return_kmeans:
        list_df_meter, \
        list_df_appliance, \
        list_df_status, kmeans = params
    else:
        list_df_meter, \
        list_df_appliance, \
        list_df_status = params

    num_buildings = len(buildings)

    # Load the data loaders
    dl_train, \
    dl_valid, \
    dl_test = datastores_to_dataloaders(list_df_meter, list_df_appliance,
                                        list_df_status,
                                        num_buildings, build_idx_train,
                                        build_idx_valid, build_idx_test,
                                        train_size=train_size,
                                        valid_size=valid_size,
                                        batch_size=batch_size,
                                        seq_len=seq_len, border=border,
                                        power_scale=power_scale)

    return_params = (dl_train, dl_valid, dl_test)
    if return_kmeans:
        return_params += (kmeans,)

    return return_params
