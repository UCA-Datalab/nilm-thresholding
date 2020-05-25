import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import preprocessing_pipeline_dict
from better_nilm.model.preprocessing import denormalize_meters

from better_nilm.model.gru import create_gru_model
from better_nilm.model.train import train_with_validation

from better_nilm.model.scores import regression_score_dict
from better_nilm.model.scores import classification_scores_dict

# This path is set to work on Zappa
dict_path_train = {"../nilm/data/nilmtk/ukdale.h5": [1, 5]}
dict_path_test = {"../nilm/data/nilmtk/ukdale.h5": 2}

appliances = ['dishwasher',
              'fridge',
              'washingmachine']

thresholds = [10,  # dishwasher
              50,  # fridge
              20]  # washingmachine

sample_period = 60  # in seconds
series_len = 510  # in number of records
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
                                 appliances=appliances,
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
                                          thresholds=thresholds,
                                          shuffle=False)

x_train = dict_prepro["train"]["x"]
y_train = dict_prepro["train"]["y"]
bin_train = dict_prepro["train"]["bin"]

x_val = dict_prepro["validation"]["x"]
y_val = dict_prepro["validation"]["y"]
bin_val = dict_prepro["validation"]["bin"]

y_max = dict_prepro["max_values"]["y"]

appliances = dict_prepro["appliances"]
num_appliances = dict_prepro["num_appliances"]

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

ser, meters = buildings_to_array(dict_path_test,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

dict_prepro = preprocessing_pipeline_dict(ser, meters,
                                          train_size=train_size,
                                          validation_size=validation_size,
                                          thresholds=thresholds,
                                          shuffle=False)

x_test = dict_prepro["test"]["x"]
y_test = dict_prepro["test"]["y"]
bin_test = dict_prepro["test"]["bin"]

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
