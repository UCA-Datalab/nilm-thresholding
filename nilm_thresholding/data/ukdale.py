import os

import pandas as pd
from pandas import Series
from pandas.io.pytables import HDFStore

from nilm_thresholding.data.preprocessing import PreprocessWrapper
from nilm_thresholding.utils.string import APPLIANCE_NAMES, homogenize_string


class UkdalePreprocess(PreprocessWrapper):
    dataset: str = "ukdale"
    datastore: HDFStore = None

    def __init__(self, path_h5: str, path_labels: str, **kwargs):
        super(UkdalePreprocess, self).__init__(**kwargs)
        self._path_h5 = path_h5
        self._path_labels = path_labels
        # Load the datastore
        self._load_datastore()

    def _load_datastore(self):
        """Loads the UKDALE h5 file as a datastore."""
        assert os.path.isfile(self._path_h5), (
            f"Input path does not lead to file:" f"\n{self._path_h5}"
        )
        assert self._path_h5.endswith(".h5"), (
            "Path must lead to a h5 file.\n" f"Input is {self._path_h5}"
        )
        self.datastore = pd.HDFStore(self._path_h5)

    def _load_meter(self, building: int, meter: int) -> pd.Series:
        """
        Loads an UKDALE meter from the datastore, and resamples it to given period.

        Parameters
        ----------
        building : int
            Building ID.
        meter : int
            Meter ID.

        Returns
        -------
        s : pandas.Series
        """
        key = "/building{}/elec/meter{}".format(building, meter)
        m = self.datastore[key]
        v = m.values.flatten()
        t = m.index
        s = pd.Series(v, index=t).clip(0.0, self.max_power)
        s[s < 10.0] = 0.0
        s = s.resample("1s").ffill(limit=300).fillna(0.0)
        s = s.resample(self.period).mean().tz_convert("UTC")
        return s

    def _datastore_to_series(self, house: int, label: str) -> pd.Series:
        """Extracts a specific label (appliance) from a house, given the datastore

        Parameters
        ----------
        house : int
            Building ID
        label : str
            Meter name

        Returns
        -------
        s : pandas.Series
        """
        # Load the meter labels
        filename = f"{self._path_labels}/house_%1d/labels.dat" % house

        labels = pd.read_csv(
            filename, delimiter=" ", header=None, index_col=0
        ).to_dict()[1]

        # Homogenize input label
        label = homogenize_string(label)
        label = APPLIANCE_NAMES.get(label, label)

        # Series placeholder
        s = None

        # Iterate through all the existing labels, searching for the input label
        for i in labels:
            lab = homogenize_string(labels[i])
            lab = APPLIANCE_NAMES.get(lab, lab)
            # When we find the input label, we load the meter records
            if lab == label:
                s = self._load_meter(house, i)

        if s is None:
            raise ValueError(
                f"Label {label} not found on house {house}\n"
                f"Valid labels are: {list(labels.values())}"
            )

        assert type(s) is Series, (
            f"load_ukdale_meter() should output {Series}\n"
            f"Received {type(s)} instead"
        )

        s.index.name = "datetime"
        s.name = label

        return s

    def load_house_meters(self, house: int) -> pd.DataFrame:
        # Make a list of meters
        list_meters = self.appliances.copy()
        list_meters.append("aggregate")

        meters = []
        for label in list_meters:
            series_meter = self._datastore_to_series(house, label)
            meters += [series_meter.rename(label)]

        meters = pd.concat(meters, axis=1)
        meters.fillna(method="pad", inplace=True)
        assert meters.shape[1] == len(list_meters), (
            f"meters dataframe must have {len(list_meters)} columns\n"
            f"It currently has {meters.shape[1]} "
        )

        # Pick range of dates
        try:
            dates = self.dates[house]
        except KeyError:
            dates = self.dates[str(house)]
        date_start = pd.to_datetime(dates[0]).tz_localize("Etc/UCT")
        date_end = pd.to_datetime(dates[1]).tz_localize("Etc/UCT")

        assert date_end > date_start, (
            f"Start date is {date_start}\nEnd date is {date_end}\n"
            "End date must be after start date!"
        )

        meters = meters[date_start:date_end]

        assert meters.shape[0] > 0, (
            "meters dataframe was left empty after applying dates\n"
            f"Start date is {date_start}\nEnd date is {date_end}"
        )

        return meters
