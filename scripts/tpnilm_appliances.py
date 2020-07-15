import os
import sys

path_main = os.path.realpath(__file__)
path_main = path_main.rsplit('/', 2)[0]
sys.path.insert(0, path_main)

"""
Trains several TP-NILM models for different appliances and buildings
"""

# Parameters to modify

path_h5 = os.path.join(path_main, 'data/ukdale.h5')
path_data = os.path.join(path_main, '../nilm/data/ukdale')

buildings = [1, 2, 5]
appliances = ['fridge', 'dish_washer', 'washing_machine']

threshold_method = 'vs'

class_w = 0
reg_w = 1

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

num_models = 5

# Other parameters (no need to modify these)

num_appliances = 1

# Model

from better_nilm.model.architecture.tpnilm import TPNILMModel

model_name = 'TPNILMModel'
model_params = {'seq_len': seq_len,
                # 'border': border,
                'out_channels': num_appliances,
                'init_features': 32,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'classification_w': class_w,
                'regression_w': reg_w}


# Run main script

path_scripts = os.path.join(path_main, 'scripts')
sys.path.insert(0, path_scripts)
import _script_appliances
