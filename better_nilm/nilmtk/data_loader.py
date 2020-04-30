from nilmtk import DataSet
from nilmtk.metergroup import MeterGroup

from better_nilm.format_utils import to_list
from better_nilm.format_utils import to_tuple
from better_nilm.nilmtk.metergroup_utils import get_good_sections


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


def metergroup_to_array(metergroup, appliances=None, sample_period=6,
                        window_size=600, max_windows=None):
    """

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
    window_size : int, default=600
        Number of consecutive records to take at once. By default is 600,
        which implies that a default time windows comprehends one hour
        worth of records (600 records x 6 seconds between each).
    max_windows : int, default=None
        Maximum number of windows to output.

    Returns
    -------
    ser : numpy.array
        shape = (meters, window_size, windows)
        - meters : The number of appliances, plus the main meter.
            They are sorted alphabetically by appliance name, excluding
            the main meter, which always comes first.
        - window_size : see Params.
        - windows : The amount of windows that could be extracted from the
            metergroup.

    """
    assert type(metergroup) is MeterGroup, f"metergroup param must be type " \
                                           f"nilmtk.metergroup.MeterGroup\n" \
                                           f"Input param is type " \
                                           f"{type(metergroup)}"

    good_sections = get_good_sections(metergroup, sample_period,
                                      window_size, max_windows=max_windows)
    return good_sections


