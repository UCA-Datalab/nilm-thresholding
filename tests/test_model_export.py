import os
import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import preprocessing_pipeline_dict

from better_nilm.model.architecture.gru import create_gru_model

from better_nilm.model.export import store_model_json
from better_nilm.model.export import store_dict_pkl
from better_nilm.model.export import load_model_json
from better_nilm.model.export import load_dict_pkl


# This path is set to work on Zappa
dict_path_buildings = {"../nilm/data/nilmtk/redd.h5": 1}

appliances = ["fridge", "microwave"]
sample_period = 6
series_len = 600
max_series = None
skip_first = None
to_int = True

train_size = .6
validation_size = .2
epochs = 1
batch_size = 64

# Weights
class_w = 1
reg_w = 1

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

dict_prepro = preprocessing_pipeline_dict(ser, meters,
                                          train_size=train_size,
                                          validation_size=validation_size)

x_train = dict_prepro["train"]["x"]
y_train = dict_prepro["train"]["y"]
bin_train = dict_prepro["train"]["bin"]

x_val = dict_prepro["validation"]["x"]
y_val = dict_prepro["validation"]["y"]
bin_val = dict_prepro["validation"]["bin"]

x_test = dict_prepro["tests"]["x"]
y_test = dict_prepro["tests"]["y"]
bin_test = dict_prepro["tests"]["bin"]

y_max = dict_prepro["max_values"]["y"]

thresholds = dict_prepro["thresholds"]
appliances = dict_prepro["appliances"]
num_appliances = dict_prepro["num_appliances"]

"""
Training
"""

model = create_gru_model(series_len, num_appliances, thresholds,
                         classification_weight=class_w,
                         regression_weight=reg_w)

model.fit(x_train, [y_train, bin_train],
          validation_data=(x_val, [y_val, bin_val]),
          epochs=epochs, batch_size=batch_size, shuffle=True)

[y_pred, bin_pred] = model.predict(x_test)

"""
Store
"""

path_output = "tests/output"
if not os.path.isdir(path_output):
    os.mkdir(path_output)

path_model = path_output + "/model.json"
store_model_json(model, path_model)

path_dic = path_output + "/model.pkl"
store_dict_pkl(dict_prepro["max_values"], path_dic)


"""
Load
"""

model = load_model_json(path_model)
dic = load_dict_pkl(path_dic)

# Lets assure its predictions are the same
[y_new, bin_new] = model.predict(x_test)

if (y_pred == y_new).all():
    print("Model store and load was succesful!")
else:
    raise ValueError("Model predictions changed after being stored and loaded")
