import numpy as np

from math import ceil
from math import floor
from sklearn.cluster import KMeans


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
    if shuffle:
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
    assert meters == sorted(set(meters)), "meters must be a sorted list of " \
                                          "non-duplicated elements"
    assert main in meters, f"'{main}' missing in meters:\n" \
                           f"{(', '.join(meters))}"

    # Locate the position of the main meter
    idx = meters.index(main)

    # Split X and Y data
    x = ser[:, :, idx].copy()
    x = np.expand_dims(x, axis=2)
    y = np.delete(ser.copy(), idx, axis=2)

    return x, y


def normalize_meters(ser, max_values=None):
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
        # Take one meter record, and sort the in ascending order
        # to ensure the first values correspond to OFF state
        meter = ser[:, :, idx].flatten()
        meter = np.sort(meter)
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


def get_thresholds(ser):
    """
    Returns the estimated thresholds that splits ON and OFF appliances states.

    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.

    Returns
    -------
    threshold : numpy.array
        shape = (num_meters,)

    """
    mean, std = _get_cluster_centroids(ser)

    # Sigma is a value between 0 and 1
    # sigma = the distance from OFF to ON at which we set the threshold
    sigma = std[:, 0] / (std.sum(axis=1))
    sigma = np.nan_to_num(sigma)

    # Add threshold
    threshold = mean[:, 0] + sigma * (mean[:, 1] - mean[:, 0])
    return threshold


def binarize(ser, thresholds):
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
    num_app = ser.shape[2]

    # Iterate through all the appliances
    for idx in range(num_app):
        mask_on = ser[:, :, idx] >= thresholds[idx]
        ser_bin[mask_on] = 1

    ser_bin = ser_bin.astype(int)

    return ser_bin


def preprocessing_pipeline_dict(ser, meters, train_size=.6, validation_size=.2,
                                main="_main", shuffle=True, random_seed=0,
                                normalize=True):
    """
    This function serves as a pipeline for preprocessing. It takes the whole
    array of data, splits it into train-validation-test, normalize its values
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
        raise ValueError(f"Train size: {train_size} is too low for the given "
                         f"amount of series: {num_series}")
    if floor(num_series * validation_size) <= 0:
        raise ValueError(f"Validation size: {validation_size} is too low for"
                         f"the given amount of series: {num_series}")

    # Split data intro train and validation+test
    ser_train, ser_test = train_test_split(ser, train_size,
                                           random_seed=random_seed,
                                           shuffle=shuffle)

    # Re-escale validation size. Split remaining data into validation and test
    validation_size /= (1 - train_size)
    ser_val, ser_test = train_test_split(ser_test, validation_size,
                                         random_seed=random_seed,
                                         shuffle=shuffle)

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
    thresholds = get_thresholds(y_train)
    bin_train = binarize(y_train, thresholds)
    bin_val = binarize(y_val, thresholds)
    bin_test = binarize(y_test, thresholds)

    # Appliance info
    appliances = meters.copy()
    appliances.remove("_main")
    num_appliances = len(appliances)

    # Include al the info into a dictionary
    dict_prepro = {"train": {"x": x_train,
                             "y": y_train,
                             "bin": bin_train},
                   "validation": {"x": x_val,
                                  "y": y_val,
                                  "bin": bin_val},
                   "test": {"x": x_test,
                            "y": y_test,
                            "bin": bin_test},
                   "max_values": {"x": x_max,
                                  "y": y_max},
                   "thresholds": thresholds,
                   "appliances": appliances,
                   "num_appliances": num_appliances}

    return dict_prepro
