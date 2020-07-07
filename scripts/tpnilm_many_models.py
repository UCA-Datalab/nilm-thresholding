import os
import sys

path_main = os.path.realpath(__file__)
path_main = path_main.rsplit('/', 2)[0]
sys.path.insert(0, path_main)

"""
Trains several TP-NILM models under the same conditions
"""

# Parameters to modify

path_h5 = os.path.join(path_main, 'data/ukdale.h5')
path_data = os.path.join(path_main, '../nilm/data/ukdale')

build_id_train = [1, 2, 5]
build_id_valid = [1]
build_id_test = [1]
appliances = ['fridge', 'dish_washer', 'washing_machine']

class_w = 1
reg_w = 0

dates = {1: ('2013-04-12', '2014-12-15'),
         2: ('2013-05-22', '2013-10-03 6:16'),
         5: ('2014-04-29', '2014-09-01')}

train_size = 0.8
valid_size = 0.1

seq_len = 480
border = 16
period = '1min'
power_scale = 2000.

batch_size = 32
learning_rate = 1.E-4
dropout = 0.1
epochs = 300
patience = 300

num_models = 50

# Other parameters (no need to modify these)

buildings = sorted(set(build_id_train + build_id_valid + build_id_test))
num_appliances = len(appliances)

# Model

from better_nilm.model.architecture.tpnilm import TPNILMModel

model_name = 'TPNILMModel'
model_params = {'seq_len': seq_len,
                'border': border,
                'out_channels': num_appliances,
                'init_features': 32,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'classification_w': class_w,
                'regression_w': reg_w}


# Run main script

path_scripts = os.path.join(path_main, 'scripts')
sys.path.insert(0, path_scripts)
import _script_many_models
