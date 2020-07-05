import numpy as np
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array
from better_nilm.model.preprocessing import train_test_split
from better_nilm.model.preprocessing import feature_target_split
from better_nilm.model.preprocessing import normalize_meters
from better_nilm.model.preprocessing import get_status_by_duration
from better_nilm.exploration_utils import print_basic_statistics
from better_nilm.exploration_utils import print_appliance_statistics


"""
This script tries to reproduce the results of Luca Massidda in his paper
Non-Intrusive Load Disaggregation by Convolutional Neural Network and 
Multilabel Classification
"""

# This path is set to work on Zappa
path_data = "../nilm/data/nilmtk/ukdale.h5"
buildings = [1, 2, 5]

appliances = ['dishwasher',
              'fridge',
              'washingmachine']

thresholds = [10,  # dishwasher
              50,  # fridge
              20]  # washingmachine

x_max = [2000]  # maximum load
y_max = [2500,  # dishwasher
         300,  # fridge
         2500]  # washingmachine

min_off = [30,  # dishwasher
           1,  # fridge
           3]  # washingmachine
min_on = [30,  # dishwasher
          1,  # fridge
          30]  # washingmachine

sample_period = 60  # in seconds
series_len = 512  # in number of records
border = 16  # borders lost after convolutions

max_series = 1800
skip_first = None
to_int = False
subtract_mean = True

train_size = .8

random_seed = 0
shuffle = True
num_appliances = len(appliances)

"""
Load the train data
"""

ser_train = []

for house in buildings:
    ser, meters = buildings_to_array({path_data: house},
                                     appliances=appliances,
                                     sample_period=sample_period,
                                     series_len=series_len,
                                     max_series=max_series,
                                     skip_first=skip_first,
                                     to_int=to_int)

    s_train, s_val = train_test_split(ser, train_size,
                                      random_seed=random_seed,
                                      shuffle=shuffle)
    ser_train += [s_train]

    # Only the first house is used for validation and tests
    if house == 1:
        ser_val, ser_test = train_test_split(s_val, .5,
                                             random_seed=random_seed,
                                             shuffle=shuffle)

# Free memory
del s_train, s_val

# Concatenate training list
ser_train = np.concatenate(ser_train)

"""
Preprocessing train
"""

print("\nBEFORE PREPROCESSING")

# Split data into X and Y
x_train, y_train = feature_target_split(ser_train, meters)

print_basic_statistics(x_train, "Train aggregate")
print_appliance_statistics(y_train, "Train", appliances)

x_val, y_val = feature_target_split(ser_val, meters)

# Get the binary meter status of each Y series
bin_train = get_status_by_duration(y_train, thresholds, min_off, min_on)
bin_val = get_status_by_duration(y_val, thresholds, min_off, min_on)

print("\nAFTER PREPROCESSING")

# Normalize
x_train, _ = normalize_meters(x_train, max_values=x_max,
                              subtract_mean=subtract_mean)
y_train, _ = normalize_meters(y_train, max_values=y_max)

x_val, _ = normalize_meters(x_val, max_values=x_max,
                            subtract_mean=subtract_mean)
y_val, _ = normalize_meters(y_val, max_values=y_max)

# Skip first and last border records of Y
y_train = y_train[:, border:-border, :]
bin_train = bin_train[:, border:-border, :]
y_val = y_val[:, border:-border, :]
bin_val = bin_val[:, border:-border, :]

"""
Statistics
"""
print_basic_statistics(x_train, "Train aggregate")
print_appliance_statistics(bin_train, "Train", appliances)

"""
Testing
"""

x_test, y_test = feature_target_split(ser_test, meters)
y_test = y_test[:, border:-border, :]

# Binarize
bin_test = get_status_by_duration(y_test, thresholds, min_off, min_on)

# Normalize
x_test, _ = normalize_meters(x_test, max_values=x_max,
                             subtract_mean=subtract_mean)
y_test, _ = normalize_meters(y_test, max_values=y_max)

"""
Statistics
"""
print_basic_statistics(x_test, "Test aggregate")
print_appliance_statistics(bin_test, "Test", appliances)
