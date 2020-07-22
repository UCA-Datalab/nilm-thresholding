import os

from better_nilm.ukdale.ukdale_data import load_dataloaders

from better_nilm._script._script_utils import get_model_scores
from better_nilm._script._script_utils import list_scores
from better_nilm._script._script_utils import get_path_plots
from better_nilm._script._script_utils import plot_store_results

from better_nilm.model.architecture.bilstm import BiLSTMModel
from better_nilm.model.architecture.tpnilm import TPNILMModel


def run_many_models(path_h5=None, path_data=None, path_main=None,
                    buildings=None, build_id_train=None, build_id_valid=None,
                    build_id_test=None, appliances=None,
                    class_w=0, reg_w=0, dates=None,
                    train_size=0, valid_size=0,
                    seq_len=480, border=16, period='1min', power_scale=2000.,
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
    # Load data

    params = load_dataloaders(path_h5, path_data, buildings, appliances,
                              build_id_train=build_id_train,
                              build_id_valid=build_id_valid,
                              build_id_test=build_id_test,
                              dates=dates, period=period,
                              train_size=train_size, valid_size=valid_size,
                              batch_size=batch_size, seq_len=seq_len,
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

    for i in range(num_models):
        print(f"\nModel {i + 1}\n")

        model = eval(model_name)(**model_params)

        # Train
        model.train_with_dataloader(dl_train, dl_valid,
                                    epochs=epochs,
                                    patience=patience)

        act_scr, pow_scr = get_model_scores(model, dl_test, power_scale,
                                            means, thresholds, appliances)

        act_scores += act_scr
        pow_scores += pow_scr

    # List scores

    scores = list_scores(appliances, act_scores, pow_scores, num_models)

    # Plot

    path_plots = get_path_plots(path_main, model_name)

    plot_store_results(path_plots, model_name, seq_len, period, class_w, reg_w,
                       threshold_method, train_size, valid_size, num_models,
                       batch_size, learning_rate, dropout, epochs, patience,
                       scores, appliances,
                       model, dl_test, power_scale, means, thresholds)
