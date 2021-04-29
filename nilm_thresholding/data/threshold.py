import logging

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from nilm_thresholding.utils.config import load_config, store_config, ConfigError
from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.string import APPLIANCE_NAMES, homogenize_string

# Power load thresholds (in watts) applied by AT thresholding
THRESHOLDS = {"dishwasher": 10.0, "fridge": 50.0, "washingmachine": 20.0}

# Time thresholds (in seconds) applied by AT thresholding
MIN_OFF = {"dishwasher": 30, "fridge": 1, "washingmachine": 3}

MIN_ON = {"dishwasher": 30, "fridge": 1, "washingmachine": 30}

MAX_POWER = {"dishwasher": 2500, "fridge": 300, "washingmachine": 2500}


class Threshold:
    thresholds: np.array = None  # (appliance, status)
    centroids: np.array = None  # (appliance, status)
    use_std: bool = False
    min_on: list = None
    min_off: list = None

    def __init__(
        self,
        appliances: list = None,
        method: str = "mp",
        num_status: int = 2,
    ):
        # Set thresholding method parameters
        self.appliances = [] if appliances is None else sorted(to_list(appliances))
        self.num_apps = len(self.appliances)
        self.method = method
        self.num_status = num_status
        # Set the default status function
        self._status_fun = self._compute_status
        self._initialize_params()

    def _initialize_params(self):
        """
        Given the method name and list of appliances,
        this function defines the necessary parameters to use the method
        """
        self.thresholds = np.zeros((self.num_apps, self.num_status))
        self.centroids = np.zeros((self.num_apps, self.num_status))
        if self.method == "vs":
            # Variance-Sensitive threshold
            self.use_std = True
        elif self.method == "mp":
            # Middle-Point threshold
            pass
        elif self.method == "at":
            # Activation-Time threshold
            thresholds = []
            min_off = []
            min_on = []
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
                thresholds += [[0, THRESHOLDS[label]]]
                min_off += [MIN_OFF[label]]
                min_on += [MIN_ON[label]]
            # AT thresholding only allows binary status
            self.num_status = 2
            self.thresholds = np.array(thresholds)
            self.centroids = self.thresholds.copy()
            self.min_off = min_off
            self.min_on = min_on
            self._status_fun = self._compute_status_by_activation_time
        elif self.method == "custom":
            # Custom method
            pass
        else:
            raise ValueError(
                f"Method {self.method} doesnt exist\n"
                f"Use one of the following: vs, mp, at"
            )

    def _compute_cluster_centroids(self, ser: np.array):
        """
        Returns the cluster centroids of a series of power load values

        Parameters
        ----------
        ser : numpy.array
            shape = (num_points)
            - num_points: Number of data values

        Returns
        -------
        tuple
            Mean and standard deviation

        """
        # We dont want to modify the original series
        ser = ser.copy()

        # Take one meter record
        kmeans = KMeans(n_clusters=self.num_status).fit(ser.reshape(-1, 1))

        # The mean of a cluster is the cluster centroid
        centroid = kmeans.cluster_centers_.reshape(self.num_status)

        # Compute the standard deviation of the points in each cluster
        labels = kmeans.labels_
        std = np.zeros(self.num_status)
        for split in range(self.num_status):
            std[split] = ser[labels == split].std()

        return centroid, std

    def _compute_thresholds(self, ser: np.array):
        """
        Returns the estimated thresholds that splits ON and OFF appliances states.

        Parameters
        ----------
        ser : numpy.array
            shape = (num_points)
            - num_points: Number of data values

        Returns
        -------
        tuple
            Thresholds and centroids
        """
        centroid, std = self._compute_cluster_centroids(ser)

        sigma = (
            np.nan_to_num(np.divide(std[:-1], std[:-1] + std[1:]))
            if self.use_std
            else np.repeat([0.5], self.num_status - 1)
        )
        threshold = np.zeros(self.num_status)
        threshold[1:] = centroid[:-1] + np.multiply(sigma, centroid[1:] - centroid[:-1])

        # Compute the new centroid of each cluster, for binary classification
        if self.num_status == 2:
            mask_on = ser >= threshold[1]
            centroid[0] = ser[~mask_on].mean()
            centroid[1] = ser[mask_on].mean()

        return threshold, centroid

    def update_appliance_threshold(self, ser: np.array, appliance: str):
        """Recomputes target appliance threshold and mean values, given its series"""
        if self.method == "at":
            logging.debug(f"No need to compute thresholds for AT method. Skipping!")
            return
        threshold, centroid = self._compute_thresholds(ser.flatten())
        idx = self.appliances.index(appliance)
        self.thresholds[idx, :] = threshold
        self.centroids[idx, :] = centroid
        logging.info(f"Appliance '{appliance}' thresholds have been updated")

    def _compute_status(self, ser: np.array) -> np.array:
        """Takes a power load series and computes the corresponding status"""
        # We compute the difference between each power load and status
        # The first positive value corresponds to the state of the appliance
        ser_bin = (
            np.argmin((ser[:, :, None] - self.thresholds[None, :, :]) >= 0, axis=2) - 1
        )
        ser_bin[ser_bin < 0] = self.num_status - 1
        return ser_bin

    @staticmethod
    def _get_status_by_activation_time(
        y: np.array, threshold: np.array, min_off, min_on
    ):
        """

        Parameters
        ----------
        y : numpy.array
            shape = (series_len)
            - series_len : Length of each time series.
        threshold : np.array
        min_off : int
        min_on : int

        Returns
        -------
        s : numpy.array
            shape = (series_len)
            With binary values indicating ON (1) and OFF (0) states.
        """
        shape_original = y.shape
        y = y.flatten().copy()

        condition = np.array(y > threshold[1])
        # Find the indices of changes in "condition"
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

    def _compute_status_by_activation_time(self, ser: np.array):

        ser_bin = Parallel(n_jobs=-1)(
            delayed(self._get_status_by_activation_time)(
                ser[:, idx],
                self.thresholds[idx],
                self.min_off[idx],
                self.min_on[idx],
            )
            for idx in range(self.num_apps)
        )
        ser_bin = np.stack(ser_bin).T
        return ser_bin

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
        ser_bin = self._status_fun(ser).astype(int)

        return ser_bin

    @property
    def config_key(self):
        """Key that contains the relevant values of the config file"""
        return f"{self.method}_{self.num_status}"

    def read_config(self, path_config: str):
        """Reads a config file and updates the thresholds accordingly"""
        config = load_config(path_config, self.config_key)
        for app_idx, app in enumerate(self.appliances):
            self.thresholds[app_idx, :] = config[app]["thresholds"]
            self.centroids[app_idx, :] = config[app]["centroids"]
        logging.debug(f"Threshold values retrieved from: {path_config}\n")

    def write_config(self, path_config: str):
        """Writes a config file with the current parameters"""
        dict_apps = {}
        for idx, app in enumerate(self.appliances):
            dict_update = {
                "thresholds": self.thresholds[idx, :].round(2).tolist(),
                "centroids": self.centroids[idx, :].round(2).tolist(),
            }
            dict_apps.update({app: dict_update})
        dict_config = {self.config_key: dict_apps}
        # Try to load the config file, if already exists
        try:
            config = load_config(path_config)
            config.update(dict_config)
        except ConfigError:
            config = dict_config
        store_config(path_config, config)
        logging.debug(f"Config stored at {path_config}\n")
