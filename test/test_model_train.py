import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import train_test_split
from better_nilm.model.preprocessing import feature_target_split

from better_nilm.model.gru import create_gru_model

# This path is set to work on Zappa
dict_path_buildings = {
    "../nilm/data/nilmtk/redd.h5": 1,
    "../nilm/data/nilmtk/ukdale.h5": 5}

appliances = None
sample_period = 6
series_len = 600
max_series = 50
skip_first = 10
to_int = True

train_size = .75
epochs = 10
batch_size = 40

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

num_appliances = len(meters) - 1

"""
Training
"""

model = create_gru_model(series_len, num_appliances)
model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size, shuffle=True)
