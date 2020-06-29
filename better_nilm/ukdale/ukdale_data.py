import numpy as np
import os
import pandas as pd

from pandas.io.pytables import HDFStore

from better_nilm.format_utils import to_list
from better_nilm.str_utils import homogenize_string
from better_nilm.model.preprocessing import get_thresholds
from better_nilm.model.preprocessing import get_status

APPLIANCE_NAMES = {
    "dish_washer": "dishwasher",
    "freezer": "fridge",
    "fridge_freezer": "fridge",
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
                      cutoff=10000.):
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
    cutoff : float, default=10000.
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
    s = pd.Series(v, index=t).clip(0., cutoff)
    s[s < 10.] = 0.
    s = s.resample('1s').ffill(limit=300).fillna(0.)
    s = s.resample(period).mean().tz_convert('UTC')
    return s


def ukdale_datastore_to_series(path_labels, datastore, house, label,
                               period='1min', cutoff=10000.,
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
    cutoff : float, default=10000.
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
            s = load_ukdale_meter(datastore, house, i, period, cutoff)

    if s is None:
        raise ValueError(f"Label {label} not found on house {house}\n"
                         f"Valid labels are: {list(labels.values())}")

    s.index.name = 'datetime'
    s.name = label

    return s


def load_ukdale_series(path_h5, path_labels, buildings, list_appliances):
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

    Returns
    -------
    ds_meter : list
        List of dataframes.
    ds_appliance : list
        List of dataframes.
    ds_status : list
        List of dataframes.
    """
    # Load datastore
    datastore = load_ukdale_datastore(path_h5)

    # Ensure both parameters are lists
    buildings = to_list(buildings)
    list_appliances = to_list(list_appliances)

    # Initialize list
    ds_meter = []
    ds_appliance = []
    ds_status = []

    for house in buildings:
        # Aggregate load
        meter = ukdale_datastore_to_series(path_labels, datastore, house,
                                           'aggregate', 10000.)
        appliances = []
        for app in list_appliances:
            a = ukdale_datastore_to_series(path_labels, datastore, house, app,
                                           10000.)
            appliances += [a]

        appliances = pd.concat(appliances, axis=1)
        appliances.fillna(method='pad', inplace=True)

        apps = np.expand_dims(appliances.values, axis=2)

        thresholds = get_thresholds(apps)
        status = get_status(apps, thresholds)
        status = status.reshape(status.shape[0], len(list_appliances))
        status = pd.DataFrame(status, columns=list_appliances)

        ds_meter.append(meter)
        ds_appliance.append(apps)
        ds_status.append(status)

    return ds_meter, ds_appliance, ds_status
