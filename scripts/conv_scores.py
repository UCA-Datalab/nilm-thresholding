import os
import sys
import typer

path_main = os.path.realpath(__file__)
path_main = path_main.rsplit('/', 2)[0]
sys.path.insert(0, path_main)

from better_nilm._script._script_many_models import run_many_models
from better_nilm.plot_output import (plot_weights, PATH_OUTPUT,
                                     DICT_MAE_LIM, F1_LIM)

"""
Parameters - They can be modified
"""

# Path to the original UKDALE h5 data, relative to the script route
PATH_H5 = os.path.join(path_main, 'data/ukdale.h5')
# Path to the folder containing meter info, relative to the script route
PATH_DATA = os.path.join(path_main, 'data/ukdale')
# List of buildings (houses)
BUILDING_TRAIN = [1, 2, 5]
BUILDING_VALID = [1]
BUILDING_TEST = [1]
# Date range for each building
DATES = {1: ('2013-04-12', '2014-12-15'),
         2: ('2013-05-22', '2013-10-03 6:16'),
         5: ('2014-04-29', '2014-09-01')}
# List of appliances
APPLIANCES = ['fridge', 'dish_washer', 'washing_machine']
CLASS_W = .1
THRESHOLD_METHOD = 'mp'
# Train and validation size, relative to 1
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
# Output sequence of the model
OUTPUT_LEN = 480
# Border added to input sequences, so that
# input_len = output_len + 2 * border
BORDER = 16
# Time period of the sequence
PERIOD = '1min'
# Power value by which we divide the load, normalizing it
POWER_SCALE = 2000.
# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1.E-4
DROPOUT = 0.1
EPOCHS = 300
PATIENCE = 300
# Number of models to train. Their scores are then normalized
NUM_MODELS = 5

"""
Do not modify the script below this point
"""


def main(path_h5: str = PATH_H5,
         path_data: str = PATH_DATA,
         path_output: str = PATH_OUTPUT,
         build_id_train=BUILDING_TRAIN,
         build_id_valid=BUILDING_VALID,
         build_id_test=BUILDING_TEST,
         dates=DATES,
         appliances=APPLIANCES,
         class_w: float = CLASS_W,
         threshold_method: str = THRESHOLD_METHOD,
         train_size: float = TRAIN_SIZE,
         valid_size: float = VALID_SIZE,
         output_len: int = OUTPUT_LEN,
         border: int = BORDER,
         period: str = PERIOD,
         power_scale: int = POWER_SCALE,
         batch_size: int = BATCH_SIZE,
         learning_rate: float = LEARNING_RATE,
         dropout: float = DROPOUT,
         epochs: int = EPOCHS,
         patience: int = PATIENCE,
         num_models: int = NUM_MODELS,
         plot_scores: bool = True,
         plot_scores_lim: bool = False):
    """
    Trains several CONV models under the same conditions
    """
    assert os.path.isdir(path_data), ('path_data must lead to folder:\n{}'
                                      .format(path_data))
    assert os.path.isfile(path_h5), ('path_h5 must lead to file_\n{}'
                                     .format(path_h5))
    reg_w = 1 - class_w
    buildings = sorted(set(build_id_train + build_id_valid + build_id_test))
    num_appliances = len(appliances)
    model_name = 'ConvModel'

    # Run main script
    print("CONV model\n")

    model_params = {'output_len': output_len,
                    'out_channels': num_appliances,
                    'init_features': 32,
                    'learning_rate': learning_rate,
                    'dropout': dropout,
                    'classification_w': class_w,
                    'regression_w': reg_w}

    run_many_models(path_h5=path_h5, path_data=path_data,
                    path_main=path_main,
                    buildings=buildings, build_id_train=build_id_train,
                    build_id_valid=build_id_valid,
                    build_id_test=build_id_test, appliances=appliances,
                    class_w=class_w, reg_w=reg_w, dates=dates,
                    train_size=train_size, valid_size=valid_size,
                    output_len=output_len, border=border, period=period,
                    power_scale=power_scale,
                    batch_size=batch_size, learning_rate=learning_rate,
                    dropout=dropout,
                    epochs=epochs, patience=patience,
                    num_models=num_models,
                    model_name=model_name, model_params=model_params,
                    threshold_method=threshold_method)

    if plot_scores:
        print('PLOT RESULTS!')
        if plot_scores_lim:
            dict_mae_lim = DICT_MAE_LIM
            f1_lim = F1_LIM
        else:
            dict_mae_lim = {}
            f1_lim = None
        for app in appliances:
            path_input = os.path.join(path_output, model_name)
            savefig = os.path.join(path_output, 'Conv_' + app + '.png')
            plot_weights(path_input, app, figsize=(4, 3), savefig=savefig,
                         dict_mae_lim=dict_mae_lim, f1_lim=f1_lim)


if __name__ == '__main__':
    typer.run(main)
