import numpy as np

from math import ceil


def train_test_split(ser, train_size, random_seed=0):
    assert 0 < train_size < 1, "Train size must be in range (0, 1)"

    # Compute the number of time series that will be used in training
    num_series = ser.shape[0]
    num_train = ceil(num_series * train_size)
    if num_train == num_series:
        raise ValueError(f"train_size {train_size} returns the 100% of series")

    # Shuffle our time series array
    np.random.seed(random_seed)
    shuffled = np.random.shuffle(ser)

    # Split the shuffled array into train and test
    ser_train = shuffled[:num_train, :, :]
    ser_test = shuffled[num_train:, :, :]

    num_new_series = ser_train.shape[0] + ser_test.shape[0]
    assert num_series == num_new_series, f"Number of time series after split" \
                                         f"{num_new_series}\ndoes not match " \
                                         f"the number before split {num_series}"

    return ser_train, ser_test


def feature_target_split(ser, meters, target="_main"):
    assert meters == sorted(set(meters)), "meters must be a sorted list of " \
                                          "non-duplicated elements"
    assert target in meters, f"'{target}' missing in meters:\n" \
                             f"{(', '.join(meters))}"

    # Locate the position of the target meter
    idx = meters.index(target)

    # Split X and Y data
    y = ser[:, :, idx].copy()
    x = ser.copy()
    x = x.delete(idx, axis=2)

    return x, y
