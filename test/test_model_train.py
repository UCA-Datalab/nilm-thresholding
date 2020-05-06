import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import train_test_split
from better_nilm.model.preprocessing import feature_target_split

from better_nilm.model.gru import create_gru_model

from better_nilm.plot_utils import comparison_plot

# This path is set to work on Zappa
dict_path_buildings = {"../nilm/data/nilmtk/redd.h5": 1}

appliances = "fridge"
sample_period = 6
series_len = 600
max_series = None
skip_first = None
to_int = True

train_size = .75
epochs = 200
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
x_test, y_test = feature_target_split(ser_test, meters)

num_appliances = len(meters) - 1

"""
Training
"""

model = create_gru_model(series_len, num_appliances)
model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size, shuffle=True)

y_pred = model.predict(x_test)

"""
Plot
"""

path_plots = "test/plots"
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

path_fig = os.path.join(path_plots, "model_train.png")

comparison_plot(y_test, y_pred, savefig=path_fig)
