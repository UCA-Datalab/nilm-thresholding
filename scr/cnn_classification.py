import sys

sys.path.insert(0, '../better_nilm')

from better_nilm.ukdale.ukdale_preprocessing import load_dataloaders
from better_nilm.model.architecture.tpnilm import PTPNetModel

from better_nilm.model.scores import classification_scores_dict

path_h5 = "data/ukdale.h5"
path_data = "../nilm/data/ukdale"

buildings = [1, 2, 5]
appliances = ['fridge', 'dish_washer', 'washing_machine']

train_buildings = [1, 5]
valid_buildings = [1]
test_buildings = [2]

train_size = 0.8
valid_size = 0.1

seq_len = 480
border = 16
max_power = 10000.
num_appliances = len(appliances)

batch_size = 32
learning_rate = 1.E-4
dropout = 0.1
epochs = 100
patience = 100

"""
Load data
"""

dl_train, \
dl_valid, \
dl_test = load_dataloaders(path_h5, path_data, buildings, appliances,
                           train_buildings, valid_buildings, test_buildings,
                           train_size=train_size, valid_size=valid_size,
                           batch_size=batch_size, seq_len=seq_len,
                           border=border, max_power=max_power)

"""
Training
"""

model = PTPNetModel(seq_len=seq_len, border=border,
                    out_channels=num_appliances,
                    init_features=32,
                    learning_rate=learning_rate, dropout=dropout)

model.train_with_dataloader(dl_train, dl_valid,
                            epochs=epochs,
                            patience=patience)

"""
Test
"""

x_true, p_true, s_true, s_hat = model.predict_loader(dl_test)
s_hat[s_hat >= .5] = 1
s_hat[s_hat < 0.5] = 0

class_scores = classification_scores_dict(s_hat, s_true, appliances)
for app, scores in class_scores.items():
    print(app, "\n", scores)

import ipdb; ipdb.set_trace()
