import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.lstm.preprocessing import train_test_split
from better_nilm.lstm.preprocessing import feature_target_split

# This path is set to work on Zappa
dict_path_buildings = {
    "../nilm/data/nilmtk/redd.h5": 1,
    "../nilm/data/nilmtk/ukdale.h5": 5}

appliances = None
sample_period = 6
series_len = 600
max_series = 50
skip_first = 10
to_int = True

train_size = .75

ser, meters = buildings_to_array(dict_path_buildings,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)
print(f"Original array shape: {ser.shape}")

ser_train, ser_test = train_test_split(ser, train_size)
print(f"Train shape: {ser_train.shape}")
print(f"Test shape: {ser_test.shape}")

x_train, y_train = feature_target_split(ser_train, meters)
print(f"X train shape: {x_train.shape}")
print(f"Y train shape: {y_train.shape}")
