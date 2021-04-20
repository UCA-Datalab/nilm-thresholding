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


class Threshold:
    thresholds: list = None
    means: np.array = None

    def __init__(
        self,
        appliances: list,
        method: str = "mp",
        use_std: bool = False,
        min_on: list = None,
        min_off: list = None,
    ):
        # Set thresholding method parameters
        self.appliances = to_list(appliances)
        self.method = method
        self.use_std = use_std
        self.min_on = min_on
        self.min_off = min_off
        self._get_threshold_params()

    def _get_threshold_params(self):
        """
        Given the method name and list of appliances,
        this function results the necessary parameters to use the method
        """

        if self.method == "vs":
            # Variance-Sensitive threshold
            self.use_std = True
        elif self.method == "mp":
            # Middle-Point threshold
            pass
        elif self.method == "at":
            # Activation-Time threshold
            self.thresholds = []
            self.min_off = []
            self.min_on = []
            for app in self.appliances:
                # Homogenize input label
                label = homogenize_string(app)
                label = APPLIANCE_NAMES.get(label, label)
                if label not in THRESHOLDS.keys():
                    msg = (
                        f"Appliance {app} has no AT info.\n"
                        f"Available appliances: {', '.join(THRESHOLDS.keys())}"
                    )
                    raise ValueError(msg)
                self.thresholds += [THRESHOLDS[label]]
                self.min_off += [MIN_OFF[label]]
                self.min_on += [MIN_ON[label]]
        elif self.method == "custom":
            # Custom method
            pass
        else:
            raise ValueError(
                f"Method {self.method} doesnt exist\n"
                f"Use one of the following: vs, mp, at"
            )

    @staticmethod
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

    def _get_thresholds(self, ser):
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
        mean : numpy.array
            shape = (num_meters,)

        """
        mean, std = self._get_cluster_centroids(ser)

        # Sigma is a value between 0 and 1
        # sigma = the distance from OFF to ON at which we set the threshold
        if self.use_std:
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

        return threshold, mean

    def get_status(self, ser):
        """

        Parameters
        ----------
        ser : numpy.array
            shape = (num_series, series_len, num_meters)
            - num_series : Amount of time series.
            - series_len : Length of each time series.
            - num_meters : Meters contained in the array.

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
                mask_on = ser[:, :, idx] > self.thresholds[idx]
                ser_bin[:, :, idx] = mask_on.astype(int)
            else:
                mask_on = ser[:, idx] > self.thresholds[idx]
                ser_bin[:, idx] = mask_on.astype(int)

        ser_bin = ser_bin.astype(int)

        return ser_bin

    @staticmethod
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

    def get_status_by_duration(self, ser):
        """

        Parameters
        ----------
        ser : numpy.array
            shape = (num_series, series_len, num_meters)
            - num_series : Amount of time series.
            - series_len : Length of each time series.
            - num_meters : Meters contained in the array.

        Returns
        -------
        ser_bin : numpy.array
            shape = (num_series, series_len, num_meters)
            With binary values indicating ON (1) and OFF (0) states.
        """
        num_apps = ser.shape[-1]
        ser_bin = ser.copy()

        for idx in range(num_apps):
            if ser.ndim == 3:
                y = ser[:, :, idx]
                ser_bin[:, :, idx] = self._get_app_status_by_duration(
                    y, self.thresholds[idx], self.min_off[idx], self.min_on[idx]
                )
            elif ser.ndim == 2:
                y = ser[:, idx]
                ser_bin[:, idx] = self._get_app_status_by_duration(
                    y, self.thresholds[idx], self.min_off[idx], self.min_on[idx]
                )

        return ser_bin

    @staticmethod
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
