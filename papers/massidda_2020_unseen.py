import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import metergroup_from_file
from better_nilm.nilmtk.data_loader import metergroup_to_dataframe
from better_nilm.nilmtk.data_loader import dataframe_to_array

from better_nilm.model.architecture.tpnilm import PTPNetModel

from better_nilm.model.scores import classification_scores_dict

from better_nilm.model.preprocessing import feature_target_split
from better_nilm.model.preprocessing import normalize_meters
from better_nilm.model.preprocessing import denormalize_meters
from better_nilm.model.preprocessing import get_status_by_duration

from better_nilm.plot_utils import plot_real_vs_prediction
from better_nilm.plot_utils import plot_load_and_state

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
timestamps = [(pd.to_datetime('2013-04-12').tz_localize('US/Eastern'),
               pd.to_datetime('2014-12-15').tz_localize('US/Eastern')),
              (pd.to_datetime('2013-05-22').tz_localize('US/Eastern'),
               pd.to_datetime('2013-10-03 06:16').tz_localize('US/Eastern')),
              (pd.to_datetime('2014-06-29').tz_localize('US/Eastern'),
               pd.to_datetime('2014-09-01').tz_localize('US/Eastern'))]

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

max_series = None
skip_first = None
to_int = False
subtract_mean = True
only_good_sections = False
ffill = 5
use_aggregate = False

train_size = .8
epochs = 100
patience = 100
batch_size = 32
learning_rate = 1.E-4
dropout = 0.1

random_seed = 0
shuffle = True
num_appliances = len(appliances)

"""
Load the train data
"""

ser_train = []
ser_val = []

for idx, building in enumerate(buildings):
    metergroup = metergroup_from_file(path_data, building,
                                      appliances=appliances)
    df = metergroup_to_dataframe(metergroup, appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len, max_series=max_series,
                                 to_int=to_int,
                                 only_good_sections=only_good_sections,
                                 ffill=ffill,
                                 use_aggregate=use_aggregate,
                                 verbose=True)

    time_start, time_end = timestamps[idx]
    df_split = df[time_start:time_end]
    # Ensure the number of rows is a multiple of series_len
    cutoff = (df_split.shape[0] // series_len) * series_len
    df_split = df_split.iloc[:cutoff, :]

    # Only the second house is used for test
    if building == 2:
        cutoff_test = cutoff * (2 - train_size)
        cutoff_test = int((cutoff_test // series_len) * series_len)

        df_split = df.iloc[:cutoff_test, :]
        ser_test, _ = dataframe_to_array(df_split, series_len)
    # First and fifth houses are used for train and validation
    else:
        s_train, meters = dataframe_to_array(df_split, series_len)
        ser_train += [s_train]

        # Compute validation split
        cutoff_val = cutoff * (1.5 - train_size / 2)
        cutoff_val = int((cutoff_val // series_len) * series_len)

        df_split = df.iloc[cutoff:cutoff_val, :]
        s_val, _ = dataframe_to_array(df_split, series_len)

        ser_val += [s_val]

# Free memory
del s_train, s_val

# Concatenate training and validation lists
ser_train = np.concatenate(ser_train)
ser_val = np.concatenate(ser_val)

"""
Preprocessing train
"""

# Split data into X and Y
x_train, y_train = feature_target_split(ser_train, meters)

x_val, y_val = feature_target_split(ser_val, meters)

# Get the binary meter status of each Y series
bin_train = get_status_by_duration(y_train, thresholds, min_off, min_on)
bin_val = get_status_by_duration(y_val, thresholds, min_off, min_on)

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
print_basic_statistics(x_train, "Train X")
print_appliance_statistics(bin_train, "Train", appliances)

"""
Training
"""

model = PTPNetModel(series_len=series_len, out_channels=num_appliances,
                    init_features=32,
                    learning_rate=learning_rate, dropout=dropout)

model.train_with_validation(x_train, y_train, bin_train,
                            x_val, y_val, bin_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            patience=patience)

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

# Prediction
bin_pred = model.predict(x_test)

# Convert torch tensors to numpy arrays
bin_pred = bin_pred.cpu().detach().numpy()

# Binarize prediction
bin_pred = get_status_by_duration(bin_pred, [.5] * 3, min_off, min_on)

"""
Statistics
"""
print_basic_statistics(x_test, "Test X")
print_appliance_statistics(bin_test, "Test", appliances)

"""
Scores
"""

class_scores = classification_scores_dict(bin_pred, bin_test, appliances)
for app, scores in class_scores.items():
    print(app, "\n", scores)

"""
Plot
"""

# Denormalize meters, if able
if not subtract_mean:
    y_test = denormalize_meters(y_test, y_max)

path_plots = "papers/plots"
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

for idx, app in enumerate(appliances):
    path_fig = os.path.join(path_plots,
                            f"massidda_unseen_{app}_classification.png")
    plot_real_vs_prediction(bin_test, bin_pred, idx=idx, num_series=4,
                            sample_period=sample_period, savefig=path_fig)

    path_fig = os.path.join(path_plots,
                            f"massidda_unseen_{app}_binarization.png")
    plot_load_and_state(y_test, bin_test, idx=idx, num_series=4,
                        sample_period=sample_period, savefig=path_fig)
