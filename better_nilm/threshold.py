from better_nilm.format_utils import to_list
from better_nilm.str_utils import APPLIANCE_NAMES
from better_nilm.str_utils import homogenize_string

# Power load thresholds (in watts) applied by AT thresholding
THRESHOLDS = {
    'dishwasher': 10.,
    'fridge': 50.,
    'washingmachine': 20.
}

# Time thresholds (in seconds) applied by AT thresholding
MIN_OFF = {
    'dishwasher': 30,
    'fridge': 1,
    'washingmachine': 3
}

MIN_ON = {
    'dishwasher': 30,
    'fridge': 1,
    'washingmachine': 30
}

MAX_POWER = {
    'dishwasher': 2500,
    'fridge': 300,
    'washingmachine': 2500
}


def _get_threshold_params(threshold_method, appliances):
    """
    Given the method name and list of appliances,
    this function outputs the necessary parameters to use the method in
    ukdale_data.load_ukdale_meter

    Parameters
    ----------
    threshold_method : str
    appliances : list

    Returns
    -------
    thresholds
    min_off
    min_on
    threshold_std

    """
    # Ensure appliances is list
    appliances = to_list(appliances)
    
    if threshold_method is 'vs':
        # Variance-Sensitive threshold
        threshold_std = True
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method is 'mp':
        # Middle-Point threshold
        threshold_std = False
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method is 'at':
        # Activation-Time threshold
        threshold_std = False
        thresholds = []
        min_off = []
        min_on = []
        for app in appliances:
            # Homogenize input label
            label = homogenize_string(app)
            label = APPLIANCE_NAMES.get(label, label)
            if label not in THRESHOLDS.keys():
                msg = f"Appliance {app} has no AT info.\n"\
                      f"Available appliances: {', '.join(THRESHOLDS.keys())}"
                raise ValueError(msg)
            thresholds += [THRESHOLDS[label]]
            min_off += [MIN_OFF[label]]
            min_on += [MIN_ON[label]]
    else:
        raise ValueError(f"Method {threshold_method} doesnt exist\n"
                         f"Use one of the following: vs, mp, at")

    return thresholds, min_off, min_on, threshold_std
