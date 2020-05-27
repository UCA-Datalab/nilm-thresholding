import numpy as np
import os
import pandas as pd

from nilmtk import DataSet
from nilmtk.metergroup import MeterGroup

from better_nilm.format_utils import to_list
from better_nilm.format_utils import to_tuple
from better_nilm.format_utils import flatten_list
from better_nilm.str_utils import homogenize_string

from better_nilm.nilmtk.metergroup_utils import APPLIANCE_NAMES
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
    else:
        appliances = to_list(appliances)

    # Check which target appliances are in the building
    # We must maintain the original name of the appliances in the
    # metergroup, but also check if that name coincides with target
    # appliances names - which may not be the same
    building_appliances = elec.label().split(", ")
    building_appliances = set([app.lower() for app in building_appliances])
    # Target appliances names are supposed to be homogenized
    # We homogenize the building names for the comparison, but then store
    # their original name in order to retrieve their meter later
    target_appliances = []
    not_found_apps = appliances.copy()
    for app in building_appliances:
        # Homogenize building app name while maintaining the original
        app_homo = homogenize_string(app)
        app_homo = APPLIANCE_NAMES.get(app_homo, app_homo)
        # If app is both in building and input list, add it to target
        if app_homo in appliances:
            target_appliances += [app]
            not_found_apps.remove(app_homo)

    # If there are no target appliances, raise error
    if len(target_appliances) == 0:
        raise ValueError(f"None of the target appliances found in house"
                         f" {building} of {path_file}\n"
                         f"Target appliances: {', '.join(appliances)}\n"
                         "Building appliances: "
                         f"{', '.join(building_appliances)}")

    # List not found appliances
    if len(not_found_apps) > 0:
        print("WARNING\nThe following appliances were not found in building"
              f"{building} of file {path_file}:\n"
              f"{', '.split(not_found_apps)}\nWe assume they were always OFF")

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


def _remove_exceed_records(df, sample_period, series_len):
    """
    Remove records from the dataframe until its number of rows is a multiple
    of series_len
    """
    # We dont want to work on the original df
    df = df.copy()

    # Count the number of records we must drop
    # We will drop records while ensuring each sequence is continuous
    # That is why we also compute te expected time delta from start to end
    # in any sequence
    exceed_records = df.shape[0] % series_len
    expected_delta = sample_period * (series_len - 1)

    # Initialize idx (to loop through the rows) and the list of dropped records
    idx = 0
    drop_idx = []

    while len(drop_idx) < exceed_records:
        stamp_start = df.index[idx]
        stamp_end = df.index[idx + series_len - 1]
        # If the delta in the sequence is not the expected, move it one step
        # and add the idx to our drop list
        # Otherwise, jump to the next sequence
        if (stamp_end - stamp_start).total_seconds() != expected_delta:
            drop_idx += [stamp_start]
            idx += 1
        else:
            idx += series_len

    # Drop the indexes from the dataframe
    df.drop(drop_idx, axis=0, inplace=True)

    assert df.shape[0] % series_len == 0, f"Number of rows in df " \
                                          f"{df.shape[0]}\nis not a multiple" \
                                          f" of series_len {series_len}"
    return df


def _ensure_continuous_series(df, sample_period, series_len):
    """
    Raise an error if any time series is not continuous.
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
                        series_len=600, max_series=None, to_int=True,
                        verbose=False):
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
    verbose : bool, default=False

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

    if verbose:
        print("Getting good sections of data.")

    good_sections = get_good_sections(metergroup, sample_period,
                                      series_len, max_series=max_series)

    if verbose:
        print("Loading dataframe containing good sections.")

    df = df_from_sections(metergroup, good_sections, sample_period)

    # The number of rows in the dataframe must be a multiple of series_len
    # If that is not the case, we have to selectively remove records from the
    # dataframe, ensuring that the amount of continuous sequences is maximal
    if df.shape[0] % series_len != 0:
        if verbose:
            print("There are too many records in the df. Deleting a few.")
        df = _remove_exceed_records(df, sample_period, series_len)

    # Ensure series are continuous
    if verbose:
        print("Ensuring each sequence is continuous.")
    _ensure_continuous_series(df, sample_period, series_len)

    # Sum contributions of appliances with the same name
    df = df.groupby(df.columns, axis=1).sum()

    # Change values to integer to reduce memory usage
    if to_int:
        df = df.astype(int)

    if verbose:
        print("Turning df to numpy array.")

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


def _ensure_same_meters(list_ser, list_meters, meters=None):
    """
    Makes sure every time series from the list_ser contains the same meters.

    Params
    ------
    list_ser : list
        Time series in numpy array format.
    list_meters : list
        Meters of each time series, sorted alphabetically.
    meters : list, default=None
        List of the meters that should have each time series, sorted
        alphabetically.
    """
    # If no final meters were provided, take all contained in the list_meters
    if meters is None:
        meters = flatten_list(list_meters)

    # Loop through each building, then loop through each of the meters that
    # should be in the building. If a meter is missing, insert it in the
    # proper position of the time series, as a record of constant 0 value.
    num_buildings = len(list_ser)
    for building in range(num_buildings):
        build_meters = list_meters[building]
        if build_meters != meters:
            build_ser = list_ser[building]
            for idx, meter in enumerate(meters):
                if meter not in build_meters:
                    build_ser = np.insert(build_ser, idx, 0, axis=2)
                    build_meters = np.insert(build_meters, idx, meter)
                    if build_ser[:, :, idx].sum() != 0:
                        raise ValueError(f"Meter {meter} wasnt added properly")
            list_ser[building] = build_ser.copy()
            list_meters[building] = build_meters.tolist()

    return list_ser, list_meters


def buildings_to_array(dict_path_buildings, appliances=None,
                       sample_period=6, series_len=600,
                       max_series=None, skip_first=None, to_int=True,
                       verbose=False):
    """
    Returns a time series numpy array containing the aggregated load for each
    meter in a subset of buildings defined by dict_path_buildings.

    Parameters
    ----------
    dict_path_buildings : dict
        {file_path : list_buildings}
        Keys are paths to nilmtk files
        Values are a list with the buildings IDs
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
        Maximum number of series to output per building.
    skip_first : int, default=None
        Number of time series to skip for each building, after having applied
        the max_series. The series are skipped in chronological order.
    to_int : bool, default=True
        If True, values are changed to integer. This reduces memory usage.
    verbose : bool, default=False

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
    assert type(dict_path_buildings) is dict, f"dict_path_buildings must be " \
                                              f"dict. Current type:\n" \
                                              f"{type(dict_path_buildings)}"

    # Ensure appliances are a list
    if appliances is not None:
        appliances = to_list(appliances)

    # Initialize list of time series and meters per building
    list_ser = []
    list_meters = []

    if (skip_first is not None) and (max_series is not None):
        assert max_series > skip_first, f"Number of max_series={max_series} " \
                                        f"must be greater than the number of" \
                                        f"skipped ones, skip_first " \
                                        f"={skip_first}"

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
                                              to_int=to_int,
                                              verbose=verbose)
            assert "_main" in meters, f"'_main' missing in meters:\n" \
                                      f"{', '.join(meters)}"

            if (skip_first is not None) and (ser.shape[0] <= skip_first):
                print(f"Building {building} from {path_file}\nreturned "
                      f"{ser.shape[0]} time series, which are less than "
                      f"the skipped amount {skip_first}"
                      f"\nThe entire building was skipped")
                continue
            elif skip_first is not None:
                list_ser += [ser[skip_first:, :, :]]
                list_meters += [meters]
            else:
                list_ser += [ser]
                list_meters += [meters]

    assert len(list_ser) > 0, "No time series in list"

    meters = flatten_list(list_meters)

    # If an appliance list was not specified, some series may be lacking some
    # meters. We fill the missing meters with 0 value (we assume those
    # appliances are always off)
    if appliances is None:
        list_ser, list_meters = _ensure_same_meters(list_ser, list_meters,
                                                    meters=meters)

    ser = np.concatenate(list_ser)
    assert ser.shape[1] == series_len, f"Length of array series " \
                                       f"{ser.shape[1]} does not match with " \
                                       f"input length {series_len}"
    assert ser.shape[2] == len(meters), f"Number of meters in array " \
                                        f"{ser.shape[2]} does not " \
                                        f"match\nwith the number of listed " \
                                        f"meters {len(meters)}"
    return ser, meters
