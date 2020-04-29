from better_nilm.nilmtk.data_loader import building_metergroup_from_file

path_file = "../data/nilmtk/redd.h5"
building = 1
appliances = ["dishwasher"]

metergroup = building_metergroup_from_file(path_file, building, appliances)
