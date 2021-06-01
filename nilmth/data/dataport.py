import itertools
import os

import pandas as pd
from pandas.io.pytables import HDFStore

from nilmth.data.preprocessing import Preprocessing


class Dataport(Preprocessing):
    dataset: str = "dataport"
    datastore: HDFStore = None
    _dict_houses: dict = {}

    def __init__(self, path_data: str, **kwargs):
        super(Dataport, self).__init__(**kwargs)
        self._path_data = path_data
        self._locate_houses()

    @property
    def _list_path_files(self):
        """List of data files"""
        # List data files
        path_files = []
        for path in os.listdir(self._path_data):
            path_subset = os.path.join(self._path_data, path)
            files = [
                os.path.join(path_subset, f)
                for f in os.listdir(path_subset)
                if f.endswith(".csv") and not f.startswith("meta")
            ]
            path_files += files
        return sorted(path_files)

    def _locate_houses(self):
        """Updates the _dict_houses dictionary, listing the file where we can find
        each house"""
        for path_file in self._list_path_files:
            df = pd.read_csv(path_file, usecols=["dataid"])
            self._dict_houses.update(
                dict(zip(df["dataid"].unique(), itertools.repeat(path_file)))
            )

    def load_house_meters(self, house: int) -> pd.DataFrame:
        pass
