import os

from better_nilm.ukdale.ukdale_data import load_dataloaders

from better_nilm._script._script_utils import get_model_scores
from better_nilm._script._script_utils import list_scores
from better_nilm._script._script_utils import generate_path_output
from better_nilm._script._script_utils import store_scores
from better_nilm._script._script_utils import store_plots

from better_nilm.model.architecture.bigru import BiGRUModel
from better_nilm.model.architecture.tpnilm import TPNILMModel


def run_individual_build_app(path_h5=None, path_data=None, path_main=None,
                             buildings=None, appliances=None,
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
    Train models for each combination of building and appliances.
    """
    # Set output path
    path_output = generate_path_output(path_main, model_name)

    # Load data

    for building in buildings:
        for appliance in appliances:

            print(f"\nHouse {building}, Appliance {appliance}\n")

            params = load_dataloaders(path_h5, path_data, building, appliance,
                                      dates=dates, period=period,
                                      train_size=train_size,
                                      valid_size=valid_size,
                                      batch_size=batch_size, output_len=output_len,
                                      border=border, power_scale=power_scale,
                                      return_means=True,
                                      threshold_method=threshold_method,
                                      threshold_std=threshold_std,
                                      thresholds=thresholds,
                                      min_off=min_off, min_on=min_on
                                      )

            dl_train, dl_valid, dl_test, kmeans = params
            thresholds, means = kmeans

            # Training
            act_scores = []
            pow_scores = []

            for i in range(num_models):
                print(f"\nModel {i + 1}\n")

                model = eval(model_name)(**model_params)

                model.train_with_dataloader(dl_train, dl_valid,
                                            epochs=epochs,
                                            patience=patience)

                act_scr, pow_scr = get_model_scores(model, dl_test,
                                                    power_scale,
                                                    means, thresholds,
                                                    [appliance],
                                                    min_off, min_on)

                act_scores += act_scr
                pow_scores += pow_scr

                # List scores

                scores = list_scores([appliance], act_scores, pow_scores,
                                     num_models)

            # Plot
            path_app = os.path.join(path_output,
                                    f"house_{building}_{appliance}")
            if not os.path.isdir(path_app):
                os.mkdir(path_app)

            store_scores(path_output, output_len, period, class_w, reg_w,
                         threshold_method, train_size, valid_size, num_models,
                         batch_size, learning_rate, dropout, epochs, patience,
                         scores, 0)

            store_plots(path_output, output_len, period, class_w, reg_w,
                        threshold_method, appliance, model, dl_test,
                        power_scale, means, thresholds, min_off, min_on)
