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
    thresholds: np.array = None
    means: np.array = None
    num_status: int = 2
    use_std: bool = False
    min_on: list = None
    min_off: list = None

    def __init__(
        self,
        appliances: list = None,
        method: str = "mp",
    ):
        # Set thresholding method parameters
        self.appliances = [] if appliances is None else to_list(appliances)
        self.method = method
        self.thresholds = np.repeat([[0, 0.01]], len(self.appliances), axis=0)
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

    def _get_cluster_centroids(self, ser):
        """
        Returns ON and OFF cluster centroids' mean and std

        Parameters
        ----------
        ser : numpy.array
            shape = (series_len, num_meters)
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

        # Initialize mean and std arrays
        mean = np.zeros((len(self.appliances), self.num_status))
        std = np.zeros((len(self.appliances), self.num_status))

        for idx in range(len(self.appliances)):
            # Take one meter record
            meter = ser[:, idx].flatten()
            meter = meter.reshape((len(meter), -1))
            kmeans = KMeans(n_clusters=self.num_status).fit(meter)

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
        threshold = np.zeros(self.num_status)

        for split in range(self.num_status):
            # Sigma is a value between 0 and 1
            # sigma = the distance from OFF to ON at which we set the threshold
            if self.use_std:
                sigma = std[:, split] / (std.sum(axis=1))
                sigma = np.nan_to_num(sigma)
            else:
                sigma = np.ones(mean.shape[0]) * 0.5

            # Add threshold
            threshold[split + 1] = mean[:, split] + sigma * (
                mean[:, (split + 1)] - mean[:, split]
            )

        # Compute the new mean of each cluster
        if self.num_status == 2:
            for idx in range(mean.shape[0]):
                # Flatten the series
                meter = ser[:, :, idx].flatten()
                mask_on = meter >= threshold[idx]
                mean[idx, 0] = meter[~mask_on].mean()
                mean[idx, 1] = meter[mask_on].mean()

        return threshold, mean

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

    def get_status(self, ser: np.array) -> np.array:
        """

        Parameters
        ----------
        ser : numpy.array
            shape = (series_len, num_meters)
            - series_len : Length of each time series.
            - num_meters : Meters contained in the array.

        Returns
        -------
        ser_bin : numpy.array
            shape = (series_len, num_meters)
            With binary values indicating ON (1) and OFF (0) states.
        """
        # We don't want to modify the original series
        ser = ser.copy()

        num_app = ser.shape[-1]
        ser_bin = np.zeros(ser.shape)
        # Iterate through all the appliances
        try:
            for idx in range(num_app):

                ser_bin[:, idx] = self._get_app_status_by_duration(
                    ser[:, idx],
                    self.thresholds[idx],
                    self.min_off[idx],
                    self.min_on[idx],
                )
        except TypeError:
            # We compute the difference between each power load and status
            # The first positive value corresponds to the state of the appliance
            ser_bin = np.argmax(
                (ser[:, :, None] - self.thresholds[None, :, :]) > 0, axis=2
            )

        ser_bin = ser_bin.astype(int)

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
