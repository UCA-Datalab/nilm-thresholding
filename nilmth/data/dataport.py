import itertools
import os

import pandas as pd
from pandas.io.pytables import HDFStore

from nilmth.data.preprocessing import Preprocessing
from nilmth.utils.logging import logger


LIST_NOT_APPLIANCES = [
    "active_record",
    "audit_2011",
    "survey_2011",
    "survey_2013",
    "survey_2014",
    "survey_2017",
    "survey_2019",
    "program_baseline",
    "program_energy_internet_demo",
    "program_shines",
]


class Dataport(Preprocessing):
    dataset: str = "dataport"
    datastore: HDFStore = None
    metadata: pd.DataFrame = None
    _dict_houses: dict = {}
    _list_path_files: list = []
    _list_path_metadata: list = []

    def __init__(self, path_data: str, **kwargs):
        super(Dataport, self).__init__(**kwargs)
        self._path_data = path_data
        # List paths and locate houses
        self._list_paths()
        self._locate_houses()
        self._read_metadata()

    def _list_paths(self):
        """List all paths to data files and metadata"""
        logger.debug(f"Listing data and metadata files in {self._path_data}")
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
        logger.debug("Done listing files")

    def _locate_houses(self):
        """Updates the _dict_houses dictionary, listing the file where we can find
        each house"""
        logger.debug("Listing the available houses IDs")
        for path_file in self._list_path_files:
            df = pd.read_csv(path_file, usecols=["dataid"])
            self._dict_houses.update(
                dict(zip(df["dataid"].unique(), itertools.repeat(path_file)))
            )
        logger.debug(f"Found {len(self._dict_houses)} available houses")

    def _read_metadata(self):
        """Reads all metadata files and stores them in metadata attribute"""
        logger.debug("Reading all the metadata")
        # List houses IDs
        list_dataid = [str(x) for x in self._dict_houses.keys()]
        # Initialize list of dataframes
        list_df = [pd.DataFrame()] * len(self._list_path_metadata)
        # Loop through all the metadata files
        for i, path_meta in enumerate(self._list_path_metadata):
            df = pd.read_csv(
                path_meta,
                parse_dates=[
                    "egauge_1min_min_time",
                    "egauge_1min_max_time",
                    "egauge_1s_min_time",
                    "egauge_1s_max_time",
                    "date_enrolled",
                    "date_withdrawn",
                ],
            ).query("dataid in @list_dataid")
            list_df[i] = df
        # Concatenate all metadata dataframes and do a bit of cleaning
        self.metadata = (
            pd.concat(list_df)
            .sort_values("dataid")
            .drop_duplicates()
            .reset_index(drop=True)
            .astype({"dataid": int})
            .replace("yes", True)
        )
        logger.debug("Done reading the metadata")

    def get_metadata(self, house: int) -> dict:
        """Returns the metadata of a house given its ID"""
        dict_house = self.metadata.query(f"dataid == {house}").to_dict("r")[0]
        # Include path to file
        dict_house.update({"path": self._dict_houses[house]})
        return dict_house

    def get_appliances(self, house: int) -> list:
        """List the available appliances of target house"""
        dict_house = self.get_metadata(house)
        appliances = [k for k, v in dict_house.items() if v is True]
        appliances = sorted(set(appliances) - set(LIST_NOT_APPLIANCES))
        return appliances

    def load_house_meters(self, house: int) -> pd.DataFrame:
        pass
