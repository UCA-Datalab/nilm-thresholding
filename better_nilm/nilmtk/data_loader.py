from nilmtk import DataSet

from better_nilm.format_utils import to_tuple


def building_metergroup_from_file(path_file, building, appliances):
    """
    Params
    ------
    path_file : str
        Path to the h5 file containing a dataset processed by nilmtk
    building : int
        Building ID to read
    appliances : list
        Appliances to extract

    Returns
    -------
    elec_meters : nilmtk.Metergroup
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

    # Check which target appliances are in the building
    building_appliances = elec.label().split(", ")
    building_appliances = set([app.lower() for app in building_appliances])
    if appliances is None:
        # Take all appliances
        return elec
    else:
        # Remove from list the appliances not in the building
        target_appliances = [app for app in appliances if
                             app in building_appliances]

    # If there are no target appliances, end function and return None
    if len(target_appliances) == 0:
        print(
            f"No target appliance found in house {building} of {path_file}")
        return None
    # If there is any target appliance, we keep going

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
    elec_meters = elec.select(instance=elec_instances)
    return elec_meters
