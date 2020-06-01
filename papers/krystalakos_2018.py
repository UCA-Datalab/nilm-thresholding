import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import preprocessing_pipeline_dict
from better_nilm.model.preprocessing import denormalize_meters

from better_nilm.model.gru import create_gru_model
from better_nilm.model.train import train_with_validation

from better_nilm.model.scores import regression_score_dict
from better_nilm.model.scores import classification_scores_dict

from better_nilm.model.preprocessing import feature_target_split
from better_nilm.model.preprocessing import normalize_meters
from better_nilm.model.preprocessing import binarize

from better_nilm.plot_utils import plot_real_vs_prediction
from better_nilm.plot_utils import plot_load_and_state

from better_nilm.model.export import store_model_json

# This path is set to work on Zappa
dict_path_train = {"../nilm/data/nilmtk/ukdale.h5": [1, 2, 4]}
dict_path_test = {"../nilm/data/nilmtk/ukdale.h5": 5}

appliance = 'fridge'

sample_period = 6  # in seconds
series_len = 100  # in number of records
max_series = None
skip_first = None
to_int = True

train_size = .8
validation_size = .1
epochs = 1000
batch_size = 64
patience = 100
learning_rate = 1e-3
sigma_c = 10

# Weights
class_w = 0
reg_w = 1

"""
Load the train data
"""

ser, meters = buildings_to_array(dict_path_train,
                                 appliances=appliance,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

"""
Preprocessing train
"""

dict_prepro = preprocessing_pipeline_dict(ser, meters,
                                          train_size=train_size,
                                          validation_size=validation_size,
                                          shuffle=False)

x_train = dict_prepro["train"]["x"]
y_train = dict_prepro["train"]["y"]
bin_train = dict_prepro["train"]["bin"]

x_val = dict_prepro["validation"]["x"]
y_val = dict_prepro["validation"]["y"]
bin_val = dict_prepro["validation"]["bin"]

x_max = dict_prepro["max_values"]["x"]
y_max = dict_prepro["max_values"]["y"]

appliances = dict_prepro["appliances"]
num_appliances = dict_prepro["num_appliances"]
thresholds = dict_prepro["thresholds"]

"""
Training
"""

model = create_gru_model(series_len, num_appliances, thresholds,
                               classification_weight=class_w,
                               regression_weight=reg_w,
                               sigma_c=sigma_c,
                               learning_rate=learning_rate)

model = train_with_validation(model,
                              x_train, [y_train, bin_train],
                              x_val, [y_val, bin_val],
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=False,
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

# Normalize
x_test, _ = normalize_meters(x_test, max_values=x_max)
y_test, _ = normalize_meters(y_test, max_values=y_max)

# Binarize
bin_test = binarize(y_test, thresholds)

# Prediction
[y_pred, bin_pred] = model.predict(x_test)
y_pred = denormalize_meters(y_pred, y_max)
bin_pred[bin_pred > .5] = 1
bin_pred[bin_pred <= 0.5] = 0

y_test = denormalize_meters(y_test, y_max)

"""
Scores
"""

reg_scores = regression_score_dict(y_pred, y_test, appliances)
print(reg_scores)

class_scores = classification_scores_dict(bin_pred, bin_test, appliances)
print(class_scores)

"""
Plot
"""

path_plots = "papers/plots"
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

path_fig = os.path.join(path_plots, f"krystalakos_{appliance}_regression.png")
plot_real_vs_prediction(y_test, y_pred, idx=0,
                        sample_period=sample_period, savefig=path_fig)

path_fig = os.path.join(path_plots, "krystalakos"
                                    f"_{appliance}_classification.png")
plot_real_vs_prediction(bin_test, -bin_pred, idx=0,
                        sample_period=sample_period, savefig=path_fig)

path_fig = os.path.join(path_plots,
                        f"krystalakos_{appliance}_binarization.png")
plot_load_and_state(y_test, bin_test, idx=0,
                    sample_period=sample_period, savefig=path_fig)

"""
Store model
"""

path_outputs = "papers/outputs"
if not os.path.isdir(path_outputs):
    os.mkdir(path_outputs)

path_model = os.path.join(path_outputs, f"krystalakos_{appliance}.json")

store_model_json(model, path_model)
