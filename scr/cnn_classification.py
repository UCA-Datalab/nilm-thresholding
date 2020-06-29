import sys

sys.path.insert(0, '../better_nilm')

from better_nilm.ukdale.ukdale_preprocessing import load_dataloaders

path_h5 = "data/ukdale.h5"
path_data = "../nilm/data/ukdale"

buildings = [2, 5]
appliances = ['fridge', 'dish_washer']

train_buildings = [2, 5]
valid_buildings = [2, 5]
test_buildings = [2, 5]

train_size = 0.8
valid_size = 0.1

batch_size = 64
seq_len = 512
border = 16
max_power = 100000.

dl_train, \
dl_valid, \
dl_test = load_dataloaders(path_h5, path_data, buildings, appliances,
                           train_buildings, valid_buildings, test_buildings,
                           train_size=train_size, valid_size=valid_size,
                           batch_size=batch_size, seq_len=seq_len,
                           border=border, max_power=max_power)

print(dl_train)
