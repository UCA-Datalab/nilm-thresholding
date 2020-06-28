import sys

sys.path.insert(0, '..')

from better_nilm.ukdale.ukdale_data import load_ukdale_series

path_h5 = "../data/ukdale.h5"
path_data = "../../nilm/data/ukdale"

buildings = [2]
appliances = ['fridge']

ds_meter, ds_appliance, ds_status = load_ukdale_series(path_h5, path_data,
                                                       buildings,
                                                       appliances)
