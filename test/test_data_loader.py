import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import metergroup_from_file
from better_nilm.nilmtk.data_loader import metergroup_to_array

# This path is set to work on Zappa
path_file = "../nilm/data/nilmtk/redd.h5"
building = 5

appliances = None
#appliances = ["dishwasher", "microwave"]

sample_period = 6
series_len = 500
max_series = None
to_int = True

metergroup = metergroup_from_file(path_file,
                                  building,
                                  appliances=appliances)

ser, meters = metergroup_to_array(metergroup,
                                  appliances=appliances,
                                  sample_period=sample_period,
                                  series_len=series_len,
                                  max_series=max_series,
                                  to_int=to_int)

print(f"Meters:\n{', '.join(meters)}")

print("Array shape should be: (num_series, series_len, num_meters)")
print(f"Expected shape: ({max_series if max_series is not None else 'any'}, "
      f"{series_len}, {len(meters)})")
print(f"Output shape:   {ser.shape}")
