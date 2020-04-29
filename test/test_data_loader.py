import sys
sys.path.insert(0, "../better_nilm")
from better_nilm.nilmtk.data_loader import building_metergroup_from_file

path_file = "../nilm/data/nilmtk/redd.h5"
building = 1
appliances = ["dishwasher", "microwave"]

metergroup = building_metergroup_from_file(path_file, building, appliances)
print(metergroup)
