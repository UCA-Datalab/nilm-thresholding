import os
import numpy as np
import pandas as pd
import sys

from itertools import product
from math import ceil

sys.path.insert(0, "../better_nilm")

from better_nilm.nilmtk.data_loader import buildings_to_array

from better_nilm.model.preprocessing import preprocessing_pipeline_dict
from better_nilm.model.preprocessing import denormalize_meters

from better_nilm.plot_utils import plot_real_vs_prediction

from better_nilm.model.gru import create_gru_model
from better_nilm.model.train import train_with_validation

from better_nilm.model.scores import regression_score_dict
from better_nilm.model.scores import classification_scores_dict

"""
Test different classification and regression weights for each of the listed 
appliances in each listed house.
"""

"""
PARAMETERS
Can be modified at will
"""

# List of appliances
appliances = ['fridge',
              'kettle',
              'microwave']

# List of buildings
dict_path_buildings = {
    "../nilm/data/nilmtk/redd.h5": [1, 2, 3, 4, 5, 6],
    "../nilm/data/nilmtk/ukdale.h5": [2, 3, 4, 5]}

# Output path
path_output = "outputs/sample_period_series_len/"

# Model parameters
sample_periods = [6, 30, 60]  # in seconds
series_lens = [60, 120, 600]  # in number of records
seconds_total = 86400 * 30  # (30 days) total time elapsed in all the series
skip_first = None  # number of series to skip
to_int = True  # change load values to integer

# Train parameters
train_size = .5
validation_size = .25
epochs = 1000
batch_size = 64
patience = 100
learning_rate = 1e-3
sigma_c = 10

# Weights
class_w = 0
reg_w = 1

# Choose random seeds (-1 = do not shuffle the data)
# We will train one model per seed, shuffling the data randomly
seeds = [1, 2, -1]

"""
SCRIPT
Do not change anything below this point unless you know pretty well what you 
are doing
"""

# Create output directory
if not os.path.isdir(path_output):
    os.mkdir(path_output)

# Turn buildings dictionary into list of tuples
# [(dataset, building_number), ...]
buildings = []
for path, ls in dict_path_buildings.items():
    for build in ls:
        buildings += [(path, build)]

# Combine both samples_periods and series_lens lists into list of tuples
list_params = list(product(sample_periods, series_lens))

# Loop through appliances, then through buildings
for app in appliances:
    for tuple_ in buildings:

        # Building name
        dataset = tuple_[0].rsplit("/", 1)[1]
        dataset = dataset.rsplit(".", 1)[0]
        building_name = dataset + str(tuple_[1])

        # Create sub-folder containing appliance + building
        path_subfolder = os.path.join(path_output,
                                      app + '_' + building_name)
        if not os.path.isdir(path_subfolder):
            os.mkdir(path_subfolder)

        print("\n==========================================================\n")
        print(f"Appliance: {app}\nBuilding: {building_name}\n")

        """
        Load the data
        """

        dict_path_building = {tuple_[0]: tuple_[1]}

        # Initialize scores dictionary
        dict_scores = {}
        dict_scores_std = {}

        # Loop through sample_period and series_len
        for sample_period, series_len in list_params:
            print("\n------------------------------------------------------\n")
            print(f"Sample period: {sample_period} seconds")
            print(f"Series length: {series_len}\n")

            # Adapt num_series to seconds_total
            num_series = ceil(seconds_total / (sample_period * series_len))

            # If the appliance is not in the building, skip it
            try:
                ser, meters = buildings_to_array(dict_path_building,
                                                 appliances=app,
                                                 sample_period=
                                                 sample_period,
                                                 series_len=series_len,
                                                 max_series=num_series,
                                                 skip_first=skip_first,
                                                 to_int=to_int)
            except ValueError:
                print("Appliance not found in building.\nSkipped.\n")
                continue

            if ser.shape[0] != num_series:
                print(f"WARNING\nDesired number of series is"
                      f" {num_series}\n"
                      f"but the amount retrieved is {ser.shape[0]}")

            if ser.shape[1] != series_len:
                raise ValueError(f"Series length must be {series_len}\n"
                                 f"Retrieved length is {ser.shape[1]}")

            # Initialize score list
            scores_values = []

            # Iterate through random seeds
            for seed in seeds:

                """
                Preprocessing
                """

                try:
                    dict_prepro = preprocessing_pipeline_dict(ser, meters,
                                                              train_size=
                                                              train_size,
                                                              validation_size=
                                                              validation_size,
                                                              shuffle=True,
                                                              random_seed=seed)
                except ValueError:
                    print("Couldn't preprocess the data due to a low amount "
                          "of series.\nSkipped.\n")
                    continue

                x_train = dict_prepro["train"]["x"]
                y_train = dict_prepro["train"]["y"]
                bin_train = dict_prepro["train"]["bin"]

                x_val = dict_prepro["validation"]["x"]
                y_val = dict_prepro["validation"]["y"]
                bin_val = dict_prepro["validation"]["bin"]

                x_test = dict_prepro["test"]["x"]
                y_test = dict_prepro["test"]["y"]
                bin_test = dict_prepro["test"]["bin"]

                x_max = dict_prepro["max_values"]["x"]
                y_max = dict_prepro["max_values"]["y"]

                apps = dict_prepro["appliances"]
                num_appliances = dict_prepro["num_appliances"]
                thresholds = dict_prepro["thresholds"]

                """
                Training
                """

                model = create_gru_model(series_len, num_appliances,
                                         thresholds,
                                         classification_weight=class_w,
                                         regression_weight=reg_w,
                                         learning_rate=learning_rate,
                                         sigma_c=sigma_c)

                model = train_with_validation(model,
                                              x_train, [y_train, bin_train],
                                              x_val, [y_val, bin_val],
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              patience=patience)

                """
                Score
                """

                [y_pred, bin_pred] = model.predict(x_test)

                # Denormalize regression and binarize classification
                y_pred[y_pred < 0] = 0
                y_pred = denormalize_meters(y_pred, y_max)
                bin_pred[bin_pred > .5] = 1
                bin_pred[bin_pred <= 0.5] = 0

                y_test_denorm = denormalize_meters(y_test, y_max)

                # Get scores dictionaries, which have the format
                # {app: {score: value, score: value, ...}}
                reg_scores = regression_score_dict(y_pred, y_test_denorm,
                                                   apps)
                class_scores = classification_scores_dict(bin_pred, bin_test,
                                                          apps)

                # Merge both dicts into a single dict with format
                # {score: value, ...}
                all_scores = reg_scores[app]
                all_scores.update(class_scores[app])

                # Add scores' values to list of values only
                # [[values_seed_0], [values_seed_1], ...]
                scores_values += [list(all_scores.values())]

            # Ensure that at least one model was trained
            if len(scores_values) <= 0:
                continue

            # Turn list of list to np array and average values
            # We get one value per metric
            scores_values = np.array(scores_values)

            scores_mean = np.mean(scores_values, axis=0)
            scores_mean = np.round(scores_mean, 4).tolist()

            scores_std = np.std(scores_values, axis=0)
            scores_std = np.round(scores_std, 4).tolist()

            # Name this combination
            name = "sample_" + str(int(sample_period)) + "_length_" + \
                   str(int(series_len))

            # Add to scores dictionary
            # {name: {score: value, ...}, ...}
            dict_scores[name] = dict(zip(all_scores.keys(), scores_mean))
            dict_scores_std[name] = dict(zip(all_scores.keys(), scores_std))

            """
            Plot
            """

            # Denormalize threshold
            threshold = int(thresholds[0] * y_max[0])

            # We will plot the last random seed, which should be the
            # un-shuffled one

            # Store test plots
            path_fig = os.path.join(path_subfolder, f"{name}_reg.png")
            plot_real_vs_prediction(y_test_denorm, y_pred, idx=0,
                                    sample_period=sample_period,
                                    savefig=path_fig, threshold=threshold)

            path_fig = os.path.join(path_subfolder, f"{name}_class.png")
            plot_real_vs_prediction(bin_test, -bin_pred, idx=0,
                                    sample_period=sample_period,
                                    savefig=path_fig)

            # Store train plot
            x_test_denorm = denormalize_meters(x_test, x_max)

            path_fig = os.path.join(path_subfolder, f"{name}_reg_total.png")
            plot_real_vs_prediction(y_test_denorm, y_pred, idx=0,
                                    sample_period=sample_period,
                                    savefig=path_fig, y_total=x_test_denorm)

        # Create dataframe from dictionary
        df = pd.DataFrame.from_dict(dict_scores)
        path_df = os.path.join(path_subfolder, "scores.csv")
        df.to_csv(path_df)

        df = pd.DataFrame.from_dict(dict_scores_std)
        path_df = os.path.join(path_subfolder, "scores_std.csv")
        df.to_csv(path_df)

        print("Done.\n")
