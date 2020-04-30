import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import metergroup_from_file
from better_nilm.nilmtk.data_loader import metergroup_to_array

# This path is set to work on Zappa
path_file = "../nilm/data/nilmtk/redd.h5"
building = 5
appliances = ["dishwasher", "microwave"]

metergroup = metergroup_from_file(path_file, building, appliances=appliances)
output = metergroup_to_array(metergroup, max_windows=50)
print(output)
