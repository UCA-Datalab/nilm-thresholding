import os
import pandas as pd

from pandas.io.pytables import HDFStore

from better_nilm.model.preprocessing import get_thresholds


def load_ukdale_datastore(path_data):
    assert os.path.isfile(path_data), f"Input path does not lead to file:" \
                                      f"\n{path_data}"
    assert path_data.endswith('.h5'), "Path must lead to a h5 file.\n" \
                                      f"Input is {path_data}"
    datastore = pd.HDFStore(path_data)
    return datastore


def resample_ukdale_meter(datastore, building=1, meter=1, period='1min',
                          cutoff=1000.):
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


def ukdale_datastore_to_series(path_data, datastore, house, label, cutoff,
                               verbose=True):
    assert os.path.isdir(path_data), "Input path is not a directory:" \
                                     f"\n{path_data}"
    filename = f"{path_data}/house_%1d/labels.dat" % house
    assert os.path.isfile(filename), f"Path not found:\n{filename}"

    if verbose:
        print(filename)

    labels = pd.read_csv(filename, delimiter=' ',
                         header=None, index_col=0).to_dict()[1]

    for i in labels:
        if labels[i] == label:
            print(i, labels[i])
            s = resample_ukdale_meter(datastore, house, i, '1min', cutoff)

    s.index.name = 'datetime'
    s.name = label

    return s


def load_ukdale_series(path_h5, path_data, buildings, appliances):
    datastore = load_ukdale_datastore(path_h5)

    ds_meter = []
    ds_appliance = []
    ds_status = []

    for house in buildings:
        # Aggregate load
        meter = ukdale_datastore_to_series(path_data, datastore, house,
                                           'aggregate', 10000.)
        apps = []
        for app in appliances:
            a = ukdale_datastore_to_series(path_data, datastore, house, app,
                                           10000.)
            apps += [a]

        apps = pd.concat(apps, axis=1)
        apps.fillna(method='pad', inplace=True)

        status = pd.DataFrame()
        for app in appliances:
            status = pd.concat([status,
                                get_thresholds(apps[app])], axis=1)

        ds_meter.append(meter)
        ds_appliance.append(apps)
        ds_status.append(status)

    return ds_meter, ds_appliance, ds_status
