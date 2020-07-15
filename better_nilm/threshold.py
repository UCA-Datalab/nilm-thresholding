from better_nilm.str_utils import APPLIANCE_NAMES
from better_nilm.str_utils import homogenize_string

THRESHOLDS = {
    'dishwasher': 10.,
    'fridge': 50.,
    'washingmachine': 20.
}

MIN_OFF = {
    'dishwasher': 30,
    'fridge': 1,
    'washingmachine': 3
}

MIN_ON = {
    'diswasher': 30,
    'fridge': 1,
    'washingmachine': 30
}


def get_threshold_method(threshold_method, appliances):
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
            thresholds += [THRESHOLDS[label]]
            min_off += [MIN_OFF[label]]
            min_on += [MIN_ON[label]]
    else:
        print(f"Method {threshold_method} doesnt exist\n"
              f"Use one of the following: vs, mp, at")

    return thresholds, min_off, min_on, threshold_std
