import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import train_test_split

from better_nilm.model.architecture.tpnilm import PTPNetModel

from better_nilm.model.scores import classification_scores_dict

from better_nilm.model.preprocessing import feature_target_split
from better_nilm.model.preprocessing import normalize_meters
from better_nilm.model.preprocessing import denormalize_meters
from better_nilm.model.preprocessing import get_status_by_duration

from better_nilm.plot_utils import plot_real_vs_prediction
from better_nilm.plot_utils import plot_load_and_state

"""
This script tries to reproduce the results of Luca Massidda in his paper
Non-Intrusive Load Disaggregation by Convolutional Neural Network and 
Multilabel Classification
"""

# This path is set to work on Zappa
dict_path_train = {"../nilm/data/nilmtk/ukdale.h5": [1, 5]}
dict_path_test = {"../nilm/data/nilmtk/ukdale.h5": 2}

appliances = ['dishwasher',
              'fridge',
              'washingmachine']

thresholds = [10,  # dishwasher
              50,  # fridge
              20]  # washingmachine

x_max = [10000]  # maximum load
y_max = [2500,  # dishwasher
         300,  # fridge
         2500]  # washingmachine

min_off = [3,  # dishwasher
           0,  # fridge
           30]  # washingmachine
min_on = [30,  # dishwasher
          1,  # fridge
          30]  # washingmachine

sample_period = 60  # in seconds
series_len = 510  # in number of records

max_series = None
skip_first = None
to_int = True
subtract_mean = False

train_size = .8
epochs = 100
patience = 100
batch_size = 32
learning_rate = 1e-4

random_seed = 0
shuffle = False
num_appliances = len(appliances)

"""
Load the train data
"""

ser, meters = buildings_to_array(dict_path_train,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

"""
Preprocessing train
"""

# Split data intro train and validation
ser_train, ser_val = train_test_split(ser, train_size,
                                      random_seed=random_seed,
                                      shuffle=shuffle)
# Use half the validation (10% of the total data)
ser_val, _ = train_test_split(ser, .5,
                              random_seed=random_seed,
                              shuffle=shuffle)

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

# Skip first and last 15 records of Y
y_train = y_train[:, 15:-15, :]
bin_train = bin_train[:, 15:-15, :]
y_val = y_val[:, 15:-15, :]
bin_val = bin_val[:, 15:-15, :]

"""
Training
"""

model = PTPNetModel(series_len=series_len, out_channels=num_appliances,
                    init_features=32,
                    learning_rate=learning_rate)

model.train_with_validation(x_train, y_train, bin_train,
                            x_val, y_val, bin_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            patience=patience)

"""
Testing
"""

ser_test, _ = buildings_to_array(dict_path_test,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

x_test, y_test = feature_target_split(ser_test, meters)
y_test = y_test[:, 15:-15, :]

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
    plot_real_vs_prediction(bin_test, -bin_pred, idx=idx, num_series=2,
                            sample_period=sample_period, savefig=path_fig)

    path_fig = os.path.join(path_plots,
                            f"massidda_unseen_{app}_binarization.png")
    plot_load_and_state(y_test, bin_test, idx=idx, num_series=2,
                        sample_period=sample_period, savefig=path_fig)
