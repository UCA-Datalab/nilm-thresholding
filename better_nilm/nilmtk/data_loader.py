import numpy as np
import os
import pandas as pd

from nilmtk import DataSet
from nilmtk.metergroup import MeterGroup

from better_nilm.format_utils import to_list
from better_nilm.format_utils import to_tuple
from better_nilm.nilmtk.metergroup_utils import get_good_sections
from better_nilm.nilmtk.metergroup_utils import df_from_sections


def metergroup_from_file(path_file, building, appliances=None):
    """
    Opens given h5 file (preprocessed by nilmtk library), goes to target
    building and outputs every electric meter related to the target appliances.

    Params
    ------
    path_file : str
        Path to the h5 file containing a dataset processed by nilmtk
    building : int
        Building ID to read
    appliances : list, default=None
        Appliances to extract. If None, extract all appliances

    Returns
    -------
    metergroup : nilmtk.metergroup.Metergroup
        List of electric meters of target appliances in target building,
        also including the main (whole house) meter
    """
    assert path_file.endswith(".h5"), "Path must lead to h5 file. " \
                                      f"Input path:\n {path_file}"
    # Load the dataset (in h5 format) from given path
    data = DataSet(path_file)

    assert building in data.buildings, f"Building {building} not in dataset " \
                                       f"buildings:\n{data.buildings}"
    # Load meter records
    elec = data.buildings.get(building).elec

    if appliances is None:
        # Take all appliances
        return elec

    # Check which target appliances are in the building
    building_appliances = elec.label().split(", ")
    building_appliances = set([app.lower() for app in building_appliances])
    # Remove from list the appliances not in the building
    target_appliances = [app for app in to_list(appliances) if
                         app in building_appliances]

    # If there are no target appliances, raise error
    if len(target_appliances) == 0:
        raise ValueError(f"None of the target appliances found in house"
                         f" {building} of {path_file}\n"
                         f"Target appliances: {', '.join(appliances)}\n"
                         "Building appliances: "
                         f"{', '.join(building_appliances)}")

    # Total electric load (aggregated)
    elec_main = to_tuple(elec.mains().instance())
    # Appliance electric load
    elec_app = elec.select_using_appliances(type=target_appliances).instance()
    # Merge both lists
    elec_instances = list(elec_main + elec_app)

    # Some elements may be tuple instead of integer
    elec_int = [i for i in elec_instances if type(i) is int]
    elec_tuples = [t for t in elec_instances if type(t) is tuple]
    elec_tuples = [i for t in elec_tuples for i in t]
    elec_instances = list(elec_int + elec_tuples)

    # Take all relevant meters
    metergroup = elec.select(instance=elec_instances)
    return metergroup


def _ensure_continuous_series(df, sample_period, series_len):
    """
    Raise an error if any time series is not continuous.

    Params
    ------
    df : pandas.DataFrame
    series_len : int
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be dates.\nCurrent type is: "
                        f"{df.index.dtype}")
    dates = df.index.values
    num_series = int(len(dates) / series_len)
    dates = np.reshape(dates, (num_series, series_len))

    # Get expected delta in seconds
    expected_delta = sample_period * (series_len - 1)
    # Get series delta in seconds
    dates_delta = (dates[:, -1] - dates[:, 0]) / np.timedelta64(1, 's')

    for idx, delta in enumerate(dates_delta):
        if delta != expected_delta:
            raise ValueError(f"Error in series {idx}.\nExpected a delta "
                             f"between begin and end of {expected_delta} "
                             f"seconds.\nGot {delta} seconds instead.")


def metergroup_to_array(metergroup, appliances=None, sample_period=6,
                        series_len=600, max_series=None, to_int=True):
    """
    Extracts a time series numpy array containing the aggregated load for each
    meter in given nilmtk.metergroup.MeterGroup object.

    Params
    ------
    metergroup : nilmtk.metergroup.Metergroup
        List of electric meters, including the main meters of the house
    appliances : list, default=None
        List of appliances to include in the array. They don't need
        to be in the metergroup - in those cases, we assume that the
        missing appliances are always turned off (load = 0).
        If None, take all the appliances in the metergroup.
    sample_period : int, default=6
        Time between consecutive electric load records, in seconds.
        By default we take 6 seconds.
    series_len : int, default=600
        Number of consecutive records to take at once. By default is 600,
        which implies that a default time series comprehends one hour
        worth of records (600 records x 6 seconds between each).
    max_series : int, default=None
        Maximum number of series to output.
    to_int : bool, default=True
        If True, values are changed to integer. This reduces memory usage.

    Returns
    -------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : The amount of series that could be extracted from the
            metergroup.
        - series_len : see Params.
        - num_meters : The number of meters (appliances + main meter).
            They are sorted alphabetically by appliance name, excluding
            the main meter, which always comes first.
    meters : list
        List of the meters in the time series, properly sorted.
    """
    assert type(metergroup) is MeterGroup, f"metergroup param must be type " \
                                           f"nilmtk.metergroup.MeterGroup\n" \
                                           f"Input param is type " \
                                           f"{type(metergroup)}"

    good_sections = get_good_sections(metergroup, sample_period,
                                      series_len, max_series=max_series)

    df = df_from_sections(metergroup, good_sections, sample_period)

    # Ensure series are continuous
    _ensure_continuous_series(df, sample_period, series_len)

    # Sum contributions of appliances with the same name
    df = df.groupby(df.columns, axis=1).sum()

    # Change values to integer to reduce memory usage
    if to_int:
        df = df.astype(int)

    if "_main" not in df.columns:
        raise ValueError("No '_main' meter contained in df columns:\n"
                         f"{', '.join(df.columns.tolist())}")

    # Initialize meter list with the main meter
    meters = ["_main"]

    # Add the appliances to the meter list
    if appliances is not None:
        meters += list(appliances)
        meters = sorted(set(meters))
        drop_apps = [app for app in df.columns if app not in meters]
        df.drop(drop_apps, axis=1, inplace=True)
    else:
        meters += df.columns.tolist()
        meters = sorted(set(meters))

    print(meters)

    # Ensure every appliance is in the dataframe
    for meter in meters:
        if meter not in df.columns:
            df[meter] = 0

    # Sort columns by name
    df = df.reindex(sorted(df.columns), axis=1)

    # Turn df into numpy array
    ser = df.values

    # Shape appropriately
    num_series = int(df.shape[0] / series_len)
    ser = np.reshape(ser, (num_series, series_len, len(meters)))

    # Ensure the reshape has been done correctly
    df_ser_diff = (ser[0, :, 0] - df.iloc[:series_len, 0])
    df_ser_diff = (df_ser_diff != 0).sum()
    assert df_ser_diff == 0, "The reshape from df to ser tensor doesn't " \
                             "output the expected tensor."

    return ser, meters


def buildings_to_array(dict_path_buildings, appliances=None,
                       sample_period=6, series_len=600,
                       max_series=None, to_int=True):
    assert type(dict_path_buildings) is dict, f"dict_path_buildings must be " \
                                              f"dict. Current type:\n" \
                                              f"{type(dict_path_buildings)}"

    # Initialize list of time series and meters per building
    list_ser = []
    list_meters = []

    for path_file, buildings in dict_path_buildings.items():
        assert os.path.isfile(path_file), f"Key '{path_file}' is not" \
                                          "a valid path."
        buildings = to_list(buildings)
        for building in buildings:
            metergroup = metergroup_from_file(path_file, building,
                                              appliances=appliances)
            ser, meters = metergroup_to_array(metergroup,
                                              appliances=appliances,
                                              sample_period=sample_period,
                                              series_len=series_len,
                                              max_series=max_series,
                                              to_int=to_int)
            assert "_main" in meters, f"'_main' missing in meters:\n" \
                                      f"{', '.join(meters)}"
            list_ser += [ser]
            list_meters += [meters]

    # Some series may be lacking some meters. We fill the missing meters
    # with 0 value (we assume those appliances are always off)
    return
