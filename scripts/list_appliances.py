import sys

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

"""
List the appliances of each nilmtk dataset
"""

dict_path_buildings = {
    "../nilm/data/nilmtk/redd.h5": [1, 2, 3, 4, 5, 6],
    "../nilm/data/nilmtk/ukdale.h5": [2, 3, 4, 5]}

appliances = None
sample_period = 6
series_len = 2
max_series = 2
skip_first = None
to_int = True

ser, meters = buildings_to_array(dict_path_buildings,
                                 appliances=appliances,
                                 sample_period=sample_period,
                                 series_len=series_len,
                                 max_series=max_series,
                                 skip_first=skip_first,
                                 to_int=to_int)

print(ser.shape)
print(meters)

"""
Current list of appliances:
'_main'
'activespeaker'
'airconditioner'
'audioamplifier'
'boiler'
'broadbandrouter'
'ceappliance'
'clothesiron'
'coffeemaker'
'computer'
'computermonitor'
'cooker'
'desktopcomputer'
'dishwasher'
'electricfurnace'
'electricoven'
'electricspaceheater'
'electricstove'
'externalharddisk'
'fridge'
'gamesconsole'
'hairdryer'
'kettle'
'laptopcomputer'
'light'
'microwave'
'modem'
'networkattachedstorage'
'projector'
'ricecooker'
'runningmachine'
'servercomputer'
'settopbox'
'smokealarm'
'sockets'
'subpanel'
'television'
'toaster'
'unknown'
'washerdryer'
'washingmachine'
'wastedisposalunit'
"""
