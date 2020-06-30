import sys

sys.path.insert(0, '../better_nilm')

from better_nilm.ukdale.ukdale_data import load_dataloaders
from better_nilm.model.architecture.tpnilm import PTPNetModel

from better_nilm.model.scores import classification_scores_dict
from better_nilm.model.preprocessing import get_status_by_duration

path_h5 = "data/ukdale.h5"
path_data = "../nilm/data/ukdale"

buildings = [1, 2, 5]
appliances = ['fridge', 'dish_washer', 'washing_machine']

thresholds = [50., 10., 20.]
min_on = [1., 30., 30.]
min_off = [1., 30., 3.]

dates = {1:('2013-04-12', '2014-12-15'),
        2: ('2013-05-22', '2013-10-03 6:16'),
        5: ('2014-04-29', '2014-09-01')}

train_size = 0.8
valid_size = 0.1

seq_len = 480
border = 16
max_power = 2000.
num_appliances = len(appliances)

batch_size = 32
learning_rate = 1.E-4
dropout = 0.1
epochs = 100
patience = 100

"""
Seen
"""

print("\nSeen\n")

build_id_train = [1, 2, 5]
build_id_valid = [1]
build_id_test = [1]

# Load data

dl_train, \
dl_valid, \
dl_test = load_dataloaders(path_h5, path_data, buildings, appliances,
                           build_id_train=build_id_train,
                           build_id_valid=build_id_valid,
                           build_id_test=build_id_test,
                           dates=dates,
                           train_size=train_size, valid_size=valid_size,
                           batch_size=batch_size, seq_len=seq_len,
                           border=border, max_power=max_power,
                          thresholds=thresholds, min_off=min_off, min_on=min_on)

# Training

model = PTPNetModel(seq_len=seq_len, border=border,
                    out_channels=num_appliances,
                    init_features=32,
                    learning_rate=learning_rate, dropout=dropout)

model.train_with_dataloader(dl_train, dl_valid,
                            epochs=epochs,
                            patience=patience)

# Test

x_true, p_true, s_true, s_hat = model.predict_loader(dl_test)

if (min_on is None) or (min_off is None):
    s_hat[s_hat >= .5] = 1
    s_hat[s_hat < 0.5] = 0
else:
    thresh = [0.5] * len(min_on)
    s_hat = get_status_by_duration(s_hat, thresh, min_off, min_on)

class_scores = classification_scores_dict(s_hat, s_true, appliances)
for app, scores in class_scores.items():
    print(app, "\n", scores)


"""
Unseen
"""

print("\nUnseen\n")

build_id_train = [1, 5]
build_id_valid = [1]
build_id_test = [2]

# Load data

dl_train, \
dl_valid, \
dl_test = load_dataloaders(path_h5, path_data, buildings, appliances,
                           build_id_train=build_id_train,
                           build_id_valid=build_id_valid,
                           build_id_test=build_id_test,
                           dates=dates,
                           train_size=train_size, valid_size=valid_size,
                           batch_size=batch_size, seq_len=seq_len,
                           border=border, max_power=max_power,
                          thresholds=thresholds, min_off=min_off, min_on=min_on)

# Training

model = PTPNetModel(seq_len=seq_len, border=border,
                    out_channels=num_appliances,
                    init_features=32,
                    learning_rate=learning_rate, dropout=dropout)

model.train_with_dataloader(dl_train, dl_valid,
                            epochs=epochs,
                            patience=patience)

# Test

x_true, p_true, s_true, s_hat = model.predict_loader(dl_test)

if (min_on is None) or (min_off is None):
    s_hat[s_hat >= .5] = 1
    s_hat[s_hat < 0.5] = 0
else:
    thresh = [0.5] * len(min_on)
    s_hat = get_status_by_duration(s_hat, thresh, min_off, min_on)

class_scores = classification_scores_dict(s_hat, s_true, appliances)
for app, scores in class_scores.items():
    print(app, "\n", scores)
