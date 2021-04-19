from math import ceil
from math import floor

import numpy as np
from sklearn.cluster import KMeans

from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.string import APPLIANCE_NAMES
from nilm_thresholding.utils.string import homogenize_string

# Power load thresholds (in watts) applied by AT thresholding
THRESHOLDS = {"dishwasher": 10.0, "fridge": 50.0, "washingmachine": 20.0}

# Time thresholds (in seconds) applied by AT thresholding
MIN_OFF = {"dishwasher": 30, "fridge": 1, "washingmachine": 3}

MIN_ON = {"dishwasher": 30, "fridge": 1, "washingmachine": 30}

MAX_POWER = {"dishwasher": 2500, "fridge": 300, "washingmachine": 2500}


def get_threshold_params(threshold_method, appliances):
    """
    Given the method name and list of appliances,
    this function results the necessary parameters to use the method in
    ukdale_data.load_ukdale_meter

    Parameters
    ----------
    threshold_method : str
    appliances : list

    Returns
    -------
    thresholds
    min_off
    min_on
    threshold_std

    """
    # Ensure appliances is list
    appliances = to_list(appliances)

    if threshold_method == "vs":
        # Variance-Sensitive threshold
        threshold_std = True
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method == "mp":
        # Middle-Point threshold
        threshold_std = False
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method == "at":
        # Activation-Time threshold
        threshold_std = False
        thresholds = []
        min_off = []
        min_on = []
        for app in appliances:
            # Homogenize input label
            label = homogenize_string(app)
            label = APPLIANCE_NAMES.get(label, label)
            if label not in THRESHOLDS.keys():
                msg = (
                    f"Appliance {app} has no AT info.\n"
                    f"Available appliances: {', '.join(THRESHOLDS.keys())}"
                )
                raise ValueError(msg)
            thresholds += [THRESHOLDS[label]]
            min_off += [MIN_OFF[label]]
            min_on += [MIN_ON[label]]
    else:
        raise ValueError(
            f"Method {threshold_method} doesnt exist\n"
            f"Use one of the following: vs, mp, at"
        )

    return thresholds, min_off, min_on, threshold_std


def train_test_split(ser, train_size, shuffle=True, random_seed=0):
    """
    Splits data array into two shuffled stacks.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    train_size : float
        Proportion of data added to the first stack
    shuffle : bool, default=True
        Shuffle the data before splitting it
    random_seed : int, default=0
    If < 0, data is not shuffled

    Returns
    -------
    ser_train : numpy.array
        shape = (num_series * train_size, series_len, num_meters)
    ser_test : numpy.array
        shape = (num_series * test_size, series_len, num_meters)
        Where test_size = 1 - train_size

    """
    assert 0 < train_size < 1, "Train size must be in range (0, 1)"

    # We don't want to modify the original series
    ser = ser.copy()

    # Compute the number of time series that will be used in training
    num_series = ser.shape[0]
    num_train = ceil(num_series * train_size)
    if num_train == num_series:
        raise ValueError(f"train_size {train_size} returns the 100% of series")

    # Shuffle our time series array
    if shuffle and random_seed > 0:
        np.random.seed(random_seed)
        np.random.shuffle(ser)

    # Split the shuffled array into train and tests
    ser_train = ser[:num_train, :, :]
    ser_test = ser[num_train:, :, :]

    num_new_series = ser_train.shape[0] + ser_test.shape[0]
    assert num_series == num_new_series, (
        f"Number of time series after split"
        f"{num_new_series}\ndoes not match "
        f"the number before split {num_series}"
    )

    return ser_train, ser_test


def feature_target_split(ser, meters, main="_main"):
    """
    Splits data array into features (X) and targets (Y).

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    meters : list
        List of meter names, sorted alphabetically and according to ser order.
        Its length must equal num_meters (see above)
    main : str, default='_main'
        Name of the main meter, the one that is used as feature (X). Must be
        contained in meters list

    Returns
    -------
    x : numpy.array
        shape = (num_series, series_len, 1)
    y : numpy.array
        shape = (num_series, series_len, num_meters - 1)

    """
    assert meters == sorted(set(meters)), (
        "meters must be a sorted list of " "non-duplicated elements"
    )
    assert main in meters, f"'{main}' missing in meters:\n" f"{(', '.join(meters))}"

    # Locate the position of the main meter
    idx = meters.index(main)

    # Split X and Y data
    x = ser[:, :, idx].copy()
    x = np.expand_dims(x, axis=2)
    y = np.delete(ser.copy(), idx, axis=2)

    return x, y


def normalize_meters(ser, max_values=None, subtract_mean=False):
    """
    Normalize the meters values for the ser data array.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    max_values : numpy.array, default=None
        shape = (num_meters, )
        Maximum value expected for each meter. If None is supplied, the array
        is created based on the given ser array.
    subtract_mean : bool, default=False
        If True, subtract the mean of each sequence, to center it around 0.

    Returns
    -------
    ser : numpy.array
        Normalized values.
    max_values : numpy.array

    """
    # We do not want to modify the original series
    ser = ser.copy()

    if max_values is not None:
        # Ensure max_values is a numpy array
        max_values = np.array(max_values)
        if len(max_values.flatten()) != ser.shape[2]:
            raise ValueError(
                f"Length of max_values array"
                f"({len(max_values.flatten())}) must be the "
                f"number of meters in the series "
                f"({ser.shape[2]})"
            )
    else:
        max_values = ser.max(axis=1).max(axis=0)

    max_values = max_values.reshape((1, 1, ser.shape[2]))
    ser = ser / max_values

    # Fill NaNs in case one max value is 0
    ser = np.nan_to_num(ser)

    if subtract_mean:
        # Make every sequence have mean 0
        ser_mean = ser.mean(axis=1)
        ser -= np.repeat(ser_mean[:, :, np.newaxis], ser.shape[1], axis=1)
        assert (ser.mean(axis=1).round(3)).sum() == 0, "Mean of sequences is" "not 0"

    return ser, max_values


def denormalize_meters(ser, max_values):
    """
    Denormalizes the values of the ser data array.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    max_values : numpy.array
        shape = (num_meters, )
        Maximum value expected for each meter.

    Returns
    -------
    ser : numpy.array
        Denormalized values.
    max_values : numpy.array

    """
    # We do not want to modify the original series
    ser = ser.copy()

    # Ensure max_values is a numpy array
    max_values = np.array(max_values)

    if len(max_values.flatten()) != ser.shape[2]:
        raise ValueError(
            f"Length of max_values array"
            f"({len(max_values.flatten())}) must be the "
            f"number of meters in the series "
            f"({ser.shape[2]})"
        )

    # Ensure proper dimensions
    max_values = max_values.reshape((1, 1, ser.shape[2]))

    ser = ser * max_values
    return ser


def _get_cluster_centroids(ser):
    """
    Returns ON and OFF cluster centroids' mean and std

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.

    Returns
    -------
    mean : numpy.array
        shape = (num_meters,)
    std : numpy.array
        shape = (num_meters,)

    """
    # We dont want to modify the original series
    ser = ser.copy()

    # Reshape in order to have one dimension per meter
    num_meters = ser.shape[2]

    # Initialize mean and std arrays
    mean = np.zeros((num_meters, 2))
    std = np.zeros((num_meters, 2))

    for idx in range(num_meters):
        # Take one meter record
        meter = ser[:, :, idx].flatten()
        meter = meter.reshape((len(meter), -1))
        kmeans = KMeans(n_clusters=2).fit(meter)

        # The mean of a cluster is the cluster centroid
        mean[idx, :] = kmeans.cluster_centers_.reshape(2)

        # Compute the standard deviation of the points in
        # each cluster
        labels = kmeans.labels_
        lab0 = meter[labels == 0]
        lab1 = meter[labels == 1]
        std[idx, 0] = lab0.std()
        std[idx, 1] = lab1.std()

    return mean, std


def get_thresholds(ser, use_std=True, return_mean=False):
    """
    Returns the estimated thresholds that splits ON and OFF appliances states.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    use_std : bool, default=True
        Consider the standard deviation of each cluster when computing the
        threshold. If not, the threshold is set in the middle point between
        cluster centroids.
    return_mean : bool, default=False
        If True, return the means as second parameter.

    Returns
    -------
    threshold : numpy.array
        shape = (num_meters,)
    mean : numpy.array
        shape = (num_meters,)
        Only returned when return_mean is True (default False)

    """
    mean, std = _get_cluster_centroids(ser)

    # Sigma is a value between 0 and 1
    # sigma = the distance from OFF to ON at which we set the threshold
    if use_std:
        sigma = std[:, 0] / (std.sum(axis=1))
        sigma = np.nan_to_num(sigma)
    else:
        sigma = np.ones(mean.shape[0]) * 0.5

    # Add threshold
    threshold = mean[:, 0] + sigma * (mean[:, 1] - mean[:, 0])

    # Compute the new mean of each cluster
    for idx in range(mean.shape[0]):
        # Flatten the series
        meter = ser[:, :, idx].flatten()
        mask_on = meter >= threshold[idx]
        mean[idx, 0] = meter[~mask_on].mean()
        mean[idx, 1] = meter[mask_on].mean()

    if return_mean:
        return threshold, np.sort(mean)
    else:
        return threshold


def get_status(ser, thresholds):
    """

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    thresholds : numpy.array
        shape = (num_meters,)

    Returns
    -------
    ser_bin : numpy.array
        shape = (num_series, series_len, num_meters)
        With binary values indicating ON (1) and OFF (0) states.
    """
    # We don't want to modify the original series
    ser = ser.copy()

    ser_bin = np.zeros(ser.shape)
    num_app = ser.shape[-1]

    # Iterate through all the appliances
    for idx in range(num_app):
        if len(ser.shape) == 3:
            mask_on = ser[:, :, idx] > thresholds[idx]
            ser_bin[:, :, idx] = mask_on.astype(int)
        else:
            mask_on = ser[:, idx] > thresholds[idx]
            ser_bin[:, idx] = mask_on.astype(int)

    ser_bin = ser_bin.astype(int)

    return ser_bin


def preprocessing_pipeline_dict(
    ser,
    meters,
    train_size=0.6,
    validation_size=0.2,
    main="_main",
    shuffle=True,
    random_seed=0,
    thresholds=None,
    normalize=True,
):
    """
    This function serves as a pipeline for preprocessing. It takes the whole
    array of data, splits it into train-validation-tests, normalize its values
    and computes the binary classification for Y data.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    meters : list
        Names of the meters contained in ser, sorted accordingly
    train_size : float, default=0.6
        Proportion of train data
    validation_size : float, default=0.2
        Proportion of validation data
    main : str, default='_main'
        Name of the main meter, that must be contained in meters list
    shuffle : bool, default=True
        Shuffles the data before splitting it
    random_seed : int, default=0
    thresholds : list, default=None
        If not provided, they are computed.
    normalize : bool, default=True
        Normalize the data. Please bear in mind that the thresholds stored
        for binarization depend on whether you have applied normalization
        or not.

    Returns
    -------
    dict_prepro : dictionary

    """

    num_series = ser.shape[0]
    if floor(num_series * train_size) <= 0:
        raise ValueError(
            f"Train size: {train_size} is too low for the given "
            f"amount of series: {num_series}"
        )
    if floor(num_series * validation_size) <= 0:
        raise ValueError(
            f"Validation size: {validation_size} is too low for "
            f"the given amount of series: {num_series}"
        )

    # Split data intro train and validation+tests
    ser_train, ser_test = train_test_split(
        ser, train_size, random_seed=random_seed, shuffle=shuffle
    )

    # Re-escale validation size. Split remaining data into validation and tests
    validation_size /= 1 - train_size
    ser_val, ser_test = train_test_split(
        ser_test, validation_size, random_seed=random_seed, shuffle=shuffle
    )

    # Split data into X and Y
    x_train, y_train = feature_target_split(ser_train, meters, main=main)

    x_val, y_val = feature_target_split(ser_val, meters, main=main)

    x_test, y_test = feature_target_split(ser_test, meters)

    # Normalize
    if normalize:
        x_train, x_max = normalize_meters(x_train)
        y_train, y_max = normalize_meters(y_train)

        x_val, _ = normalize_meters(x_val, max_values=x_max)
        y_val, _ = normalize_meters(y_val, max_values=y_max)

        x_test, _ = normalize_meters(x_test, max_values=x_max)
        y_test, _ = normalize_meters(y_test, max_values=y_max)
    else:
        x_max = None
        y_max = None

    # Get the binary meter status of each Y series
    if thresholds is None:
        thresholds = get_thresholds(y_train)
    bin_train = get_status(y_train, thresholds)
    bin_val = get_status(y_val, thresholds)
    bin_test = get_status(y_test, thresholds)

    # Appliance info
    appliances = meters.copy()
    appliances.remove("_main")
    num_appliances = len(appliances)

    # Include al the info into a dictionary
    dict_prepro = {
        "train": {"x": x_train, "y": y_train, "bin": bin_train},
        "validation": {"x": x_val, "y": y_val, "bin": bin_val},
        "tests": {"x": x_test, "y": y_test, "bin": bin_test},
        "max_values": {"x": x_max, "y": y_max},
        "thresholds": thresholds,
        "appliances": appliances,
        "num_appliances": num_appliances,
    }

    return dict_prepro


def _get_app_status_by_duration(y, threshold, min_off, min_on):
    """

    Parameters
    ----------
    y : numpy.array
        shape = (num_series, series_len)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
    threshold : float
    min_off : int
    min_on : int

    Returns
    -------
    s : numpy.array
        shape = (num_series, series_len)
        With binary values indicating ON (1) and OFF (0) states.
    """
    shape_original = y.shape
    y = y.flatten().copy()

    condition = y > threshold
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx = d.nonzero()[0]

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    on_events = idx[:, 0].copy()
    off_events = idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000.0)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        assert len(on_events) == len(off_events)

        on_duration = off_events - on_events
        on_events = on_events[on_duration > min_on]
        off_events = off_events[on_duration > min_on]

    s = y.copy()
    s[:] = 0.0

    for on, off in zip(on_events, off_events):
        s[on:off] = 1.0

    s = np.reshape(s, shape_original)

    return s


def get_status_by_duration(ser, thresholds, min_off, min_on):
    """

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    thresholds : numpy.array
        shape = (num_meters,)
    min_off : numpy.array
        shape = (num_meters,)
    min_on : numpy.array
        shape = (num_meters,)

    Returns
    -------
    ser_bin : numpy.array
        shape = (num_series, series_len, num_meters)
        With binary values indicating ON (1) and OFF (0) states.
    """
    num_apps = ser.shape[-1]
    ser_bin = ser.copy()

    msg = (
        f"Length of thresholds ({len(thresholds)})\n"
        f"and number of appliances ({num_apps}) doesn't match\n"
    )
    assert len(thresholds) == num_apps, msg

    msg = (
        f"Length of thresholds ({len(thresholds)})\n"
        f"and min_on ({len(min_on)}) doesn't match\n"
    )
    assert len(thresholds) == len(min_on), msg

    msg = (
        f"Length of thresholds ({len(thresholds)})\n"
        f"and min_off ({len(min_off)}) doesn't match\n"
    )
    assert len(thresholds) == len(min_off), msg

    for idx in range(num_apps):
        if ser.ndim == 3:
            y = ser[:, :, idx]
            ser_bin[:, :, idx] = _get_app_status_by_duration(
                y, thresholds[idx], min_off[idx], min_on[idx]
            )
        elif ser.ndim == 2:
            y = ser[:, idx]
            ser_bin[:, idx] = _get_app_status_by_duration(
                y, thresholds[idx], min_off[idx], min_on[idx]
            )

    return ser_bin


def get_status_means(ser, status):
    """
    Get means of both status.
    """

    means = np.zeros((ser.shape[2], 2))

    # Compute the new mean of each cluster
    for idx in range(ser.shape[2]):
        # Flatten the series
        meter = ser[:, :, idx].flatten()
        mask_on = status[:, :, idx].flatten() > 0
        means[idx, 0] = meter[~mask_on].mean()
        means[idx, 1] = meter[mask_on].mean()

    return means
