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
