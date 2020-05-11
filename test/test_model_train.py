import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import train_test_split
from better_nilm.model.preprocessing import feature_target_split
from better_nilm.model.preprocessing import normalize_meters
from better_nilm.model.preprocessing import denormalize_meters
from better_nilm.model.preprocessing import get_thresholds
from better_nilm.model.preprocessing import binarize

from better_nilm.model.gru import create_gru_model

from better_nilm.plot_utils import plot_real_vs_prediction
from better_nilm.plot_utils import plot_load_and_state

# This path is set to work on Zappa
dict_path_buildings = {"../nilm/data/nilmtk/redd.h5": 1}

appliances = ["fridge", "microwave"]
sample_period = 6
series_len = 600
max_series = None
skip_first = None
to_int = True

train_size = .75
epochs = 10
batch_size = 64

"""
Load the data
"""

ser, meters = buildings_to_array(dict_path_buildings,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

"""
Preprocessing
"""

ser_train, ser_test = train_test_split(ser, train_size)

x_train, y_train = feature_target_split(ser_train, meters)
x_train, x_max = normalize_meters(x_train)
y_train, y_max = normalize_meters(y_train)

x_test, y_test = feature_target_split(ser_test, meters)
x_test, _ = normalize_meters(x_test, max_values=x_max)

thresholds = get_thresholds(y_train)
bin_train = binarize(y_train, thresholds)
bin_test = binarize(y_test, thresholds)

appliances = meters.copy()
appliances.remove("_main")
num_appliances = len(appliances)

"""
Training
"""

model = create_gru_model(series_len, num_appliances, thresholds)
model.fit(x_train, [y_train, bin_train],
          epochs=epochs, batch_size=batch_size, shuffle=True)

[y_pred, bin_pred] = model.predict(x_test)
y_pred = denormalize_meters(y_pred, y_max)
print(y_pred)

"""
Plot
"""

path_plots = "test/plots"
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

for idx, app in enumerate(appliances):
    path_fig = os.path.join(path_plots, f"model_train_{app}.png")
    plot_real_vs_prediction(y_test, y_pred, idx=idx, savefig=path_fig)

    path_fig = os.path.join(path_plots, f"model_train_{app}_bin.png")
    plot_real_vs_prediction(bin_test, bin_pred, idx=idx, savefig=path_fig)

    path_fig = os.path.join(path_plots, f"model_train_{app}_real.png")
    plot_load_and_state(y_test, bin_test, idx=idx, savefig=path_fig)
