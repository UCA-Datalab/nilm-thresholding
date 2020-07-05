from nilmtk.dataset_converters.ukdale.convert_ukdale import convert_ukdale

"""
Converts UK-DALE data to nilmtk format
"""

ukdale_path = '../nilm/data/ukdale'
output_filename = 'data/ukdale.h5'

convert_ukdale(ukdale_path, output_filename, format='HDF', drop_duplicates=True)