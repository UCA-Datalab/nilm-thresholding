import time
from collections import defaultdict

from better_nilm._script._script_utils import generate_path_output
from better_nilm._script._script_utils import get_model_scores
from better_nilm._script._script_utils import list_scores
from better_nilm._script._script_utils import store_plots
from better_nilm._script._script_utils import store_scores
from better_nilm.ukdale.ukdale_data import load_dataloaders

from better_nilm.model.architecture.conv import ConvModel
from better_nilm.model.architecture.gru import GRUModel

"""
Train the same architecture many times, using random weight initializations
"""


def _merge_dict_list(dict_list):
    d = defaultdict(dict)
    for l in dict_list:
        for elem in l:
            d[elem].update(l[elem])

    return d


def run_many_models(path_h5=None, path_data=None, path_main=None,
                    buildings=None, build_id_train=None, build_id_valid=None,
                    build_id_test=None, appliances=None,
                    class_w=0, reg_w=0, dates=None,
                    train_size=0, valid_size=0,
                    output_len=480, border=16, period='1min',
                    power_scale=2000.,
                    batch_size=32, learning_rate=0, dropout=0,
                    epochs=1, patience=1, num_models=1,
                    model_name=None, model_params=None,
                    threshold_method='custom',
                    threshold_std=True, thresholds=None,
                    min_off=None, min_on=None):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """

    # Set output path
    path_output = generate_path_output(path_main, model_name)

    # Load data

    params = load_dataloaders(path_h5, path_data, buildings, appliances,
                              build_id_train=build_id_train,
                              build_id_valid=build_id_valid,
                              build_id_test=build_id_test,
                              dates=dates, period=period,
                              train_size=train_size, valid_size=valid_size,
                              batch_size=batch_size, output_len=output_len,
                              border=border, power_scale=power_scale,
                              return_means=True,
                              threshold_method=threshold_method,
                              threshold_std=threshold_std,
                              thresholds=thresholds,
                              min_off=min_off, min_on=min_on)

    dl_train, dl_valid, dl_test, kmeans = params
    thresholds, means = kmeans

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0

    for i in range(num_models):
        print(f"\nModel {i + 1}\n")

        model = eval(model_name)(**model_params)

        # Train
        time_start = time.time()
        model.train_with_dataloader(dl_train, dl_valid,
                                    epochs=epochs,
                                    patience=patience)
        time_ellapsed += time.time() - time_start

        act_scr, pow_scr = get_model_scores(model, dl_test, power_scale,
                                            means, thresholds, appliances,
                                            min_off, min_on)

        act_scores += act_scr
        pow_scores += pow_scr

        # Store individual scores
        act_dict = _merge_dict_list(act_scr)
        pow_dict = _merge_dict_list(pow_scr)

        scores = {'classification': act_dict,
                  'regression': pow_dict}

        filename = "scores_{}.txt".format(i)

        store_scores(path_output, output_len, period, class_w, reg_w,
                     threshold_method, train_size, valid_size, num_models,
                     batch_size, learning_rate, dropout, epochs, patience,
                     scores, time_ellapsed, filename=filename)

    # List scores

    scores = list_scores(appliances, act_scores, pow_scores, num_models)

    time_ellapsed /= num_models

    # Store scores and plot

    store_scores(path_output, output_len, period, class_w, reg_w,
                 threshold_method, train_size, valid_size, num_models,
                 batch_size, learning_rate, dropout, epochs, patience,
                 scores, time_ellapsed)

    store_plots(path_output, output_len, period, class_w, reg_w,
                threshold_method, appliances, model, dl_test,
                power_scale, means, thresholds, min_off, min_on)
