import numpy as np

from math import ceil
from sklearn.cluster import KMeans


def train_test_split(ser, train_size, random_seed=0):
    assert 0 < train_size < 1, "Train size must be in range (0, 1)"

    # We don't want to modify the original series
    ser = ser.copy()

    # Compute the number of time series that will be used in training
    num_series = ser.shape[0]
    num_train = ceil(num_series * train_size)
    if num_train == num_series:
        raise ValueError(f"train_size {train_size} returns the 100% of series")

    # Shuffle our time series array
    np.random.seed(random_seed)
    np.random.shuffle(ser)

    # Split the shuffled array into train and test
    ser_train = ser[:num_train, :, :]
    ser_test = ser[num_train:, :, :]

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
    x = ser[:, :, idx].copy()
    x = np.expand_dims(x, axis=2)
    y = np.delete(ser.copy(), idx, axis=2)

    return x, y


def normalize_meters(ser, max_values=None):
    # We do not want to modify the original series
    ser = ser.copy()

    if max_values is not None:
        # Ensure max_values is a numpy array
        max_values = np.array(max_values)
        if len(max_values.flatten()) != ser.shape[2]:
            raise ValueError(f"Length of max_values array"
                             f"({len(max_values.flatten())}) must be the "
                             f"number of meters in the series "
                             f"({ser.shape[2]})")
    else:
        max_values = ser.max(axis=1).max(axis=0)

    max_values = max_values.reshape((1, 1, ser.shape[2]))
    ser = ser / max_values
    
    # Fill NaNs in case one max value is 0
    ser = np.nan_to_num(ser)
    
    return ser, max_values


def denormalize_meters(ser, max_values):
    # We do not want to modify the original series
    ser = ser.copy()

    # Ensure max_values is a numpy array
    max_values = np.array(max_values)

    if len(max_values.flatten()) != ser.shape[2]:
        raise ValueError(f"Length of max_values array"
                         f"({len(max_values.flatten())}) must be the "
                         f"number of meters in the series "
                         f"({ser.shape[2]})")

    # Ensure proper dimensions
    max_values = max_values.reshape((1, 1, ser.shape[2]))

    ser = ser * max_values
    return ser


def _get_cluster_centroids(ser):
    """
    Returns ON and OFF cluster centroids' mean and std
    """
    # We dont want to modify the original series
    ser = ser.copy()

    # Reshape in order to have one dimension per meter
    num_meters = ser.shape[2]

    # Initialize center list
    centers = []
    for idx in range(num_meters):
        meter = ser[:, :, idx]
        kmeans = KMeans(n_clusters=2).fit(meter)
        cc = kmeans.cluster_centers_
        # Ensure the lesser value goes first
        cc = np.sort(cc, axis=0)
        centers += [cc]
    
    centers = np.array(centers)
    mean = centers.mean(axis=2)
    std = centers.std(axis=2)
    
    return mean, std


def get_thresholds(ser):
    mean, std = _get_cluster_centroids(ser)

    # Sigma is a value between 0 and 1
    # sigma = the distance from OFF to ON at which we set the threshold
    sigma = std[:, 0] / (std.sum(axis=1))
    sigma = np.nan_to_num(sigma)
    
    # Add threshold
    threshold = mean[:, 0] * (1 - sigma) + mean[:, 1] * sigma
    return threshold


def binarize(ser, thresholds):
    # We dont want to modify the original series
    ser = ser.copy()
    
    ser = ser >= thresholds
    return ser.astype(int)
