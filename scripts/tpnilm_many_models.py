import os
import sys

path_main = os.path.realpath(__file__)
path_main = path_main.rsplit('/', 2)[0]
sys.path.insert(0, path_main)

"""
Trains several TP-NILM models under the same conditions
"""

# PARAMETERS TO MODIFY

# Path to the original UKDALE h5 data, relative to the script route
path_h5 = os.path.join(path_main, 'data/ukdale.h5')
# Path to the folder containing meter info, relative to the script route
path_data = os.path.join(path_main, '../nilm/data/ukdale')

# List of buildings (houses)
build_id_train = [1, 2, 5]
build_id_valid = [1]
build_id_test = [1]
# List of appliances
appliances = ['fridge', 'dish_washer', 'washing_machine']

# Threshold and weights combination
# thresh:[(clas, reg)]
w_clas = [.1, .3, .5, .7, .9]
w_reg = [round(1 - w, 2) for w in w_clas]
dict_thresh_weights = {'mp': list(zip(w_clas, w_reg)),
                     'vs': [],
                     'at': []}

# Date range for each building
dates = {1: ('2013-04-12', '2014-12-15'),
         2: ('2013-05-22', '2013-10-03 6:16'),
         5: ('2014-04-29', '2014-09-01')}

# Train and validation size, relative to 1
train_size = 0.8
valid_size = 0.1

# Output sequence of the model
output_len = 480
# Border added to input sequences, so that
# input_len = output_len + 2 * border
border = 16
# Time period of the sequence
period = '1min'
# Power value by which we divide the load, normalizing it
power_scale = 2000.

# Training parameters
batch_size = 32
learning_rate = 1.E-4
dropout = 0.1
epochs = 300
patience = 300

# Number of models to train. Their scores are then normalized
num_models = 5

# OTHER PARAMETERS (no need to modify these)

buildings = sorted(set(build_id_train + build_id_valid + build_id_test))
num_appliances = len(appliances)

# MODEL

model_name = 'TPNILMModel'

# Run main script

print("TP-NILM many models\n")

sys.path.insert(0, path_main)

from better_nilm._script._script_many_models import run_many_models

for threshold_method, weights in dict_thresh_weights.items():
    if len(weights) == 0:
        continue
    print(f"\n++++++++++++\n{threshold_method}\n+++++++++++++")
    for class_w, reg_w in weights:
        
        model_params = {'output_len': output_len,
            # 'border': border,
            'out_channels': num_appliances,
            'init_features': 32,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'classification_w': class_w,
            'regression_w': reg_w}
        
        run_many_models(path_h5=path_h5, path_data=path_data, path_main=path_main,
                        buildings=buildings, build_id_train=build_id_train,
                        build_id_valid=build_id_valid,
                        build_id_test=build_id_test, appliances=appliances,
                        class_w=class_w, reg_w=reg_w, dates=dates,
                        train_size=train_size, valid_size=valid_size,
                        output_len=output_len, border=border, period=period,
                        power_scale=power_scale,
                        batch_size=batch_size, learning_rate=learning_rate,
                        dropout=dropout,
                        epochs=epochs, patience=patience, num_models=num_models,
                        model_name=model_name, model_params=model_params,
                        threshold_method=threshold_method)
