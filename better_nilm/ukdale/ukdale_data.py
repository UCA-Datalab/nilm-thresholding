import os
import pandas as pd

from pandas.io.pytables import HDFStore


def load_ukdale_datastore(path_data):
    assert os.path.isfile(path_data), f"Input path is not a file:\n{path_data}"
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


def get_ukdale_series(datastore, house, label, cutoff, path_data,
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

    return s
