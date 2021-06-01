import itertools
import os

import pandas as pd
from pandas.io.pytables import HDFStore

from nilmth.data.preprocessing import Preprocessing


class Dataport(Preprocessing):
    dataset: str = "dataport"
    datastore: HDFStore = None
    _dict_houses: dict = {}
    _list_path_files: list = []
    _list_path_metadata: list = []

    def __init__(self, path_data: str, **kwargs):
        super(Dataport, self).__init__(**kwargs)
        self._path_data = path_data
        # List paths and locate houses
        self._list_paths()
        self._locate_houses()

    def _list_paths(self):
        """List all paths to data files and metadata"""
        list_path_files = []
        list_path_metadata = []
        # List data files
        for path in os.listdir(self._path_data):
            path_subset = os.path.join(self._path_data, path)
            files = [
                os.path.join(path_subset, f)
                for f in os.listdir(path_subset)
                if f.endswith(".csv") and not f.startswith("meta")
            ]
            list_path_files += files
            metadata = [
                os.path.join(path_subset, f)
                for f in os.listdir(path_subset)
                if f.endswith(".csv") and f.startswith("meta")
            ]
            list_path_metadata += metadata
        self._list_path_files = sorted(list_path_files)
        self._list_path_metadata = sorted(list_path_metadata)

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
