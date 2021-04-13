# Shut Future Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import typer

from better_nilm.data.ukdale import load_dataloaders
from better_nilm.model.model import initialize_model
from better_nilm.results.plot_output import plot_scores_by_class_weight
from better_nilm.results.store_output import (
    generate_path_output,
    generate_folder_name,
    get_model_scores,
    list_scores,
    store_plots,
    store_scores,
    store_real_data_and_predictions,
)
from better_nilm.utils.conf import load_conf_full, update_config
from better_nilm.utils.format_list import merge_dict_list


def test_many_models(
    path_h5,
    path_data,
    path_output,
    config: dict,
    save_scores: bool = True,
    save_predictions: bool = True,
):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """

    # Set output path
    path_output = generate_path_output(path_output, config["train"]["name"])
    path_output_folder = generate_folder_name(
        path_output,
        config["train"]["model"]["output_len"],
        config["data"]["period"],
        config["train"]["model"]["classification_w"],
        config["train"]["model"]["regression_w"],
        config["data"]["threshold"]["method"],
    )

    # Load data

    params = load_dataloaders(path_h5, path_data, config)

    dl_train, dl_valid, dl_test, kmeans = params
    thresholds, means = kmeans

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0

    for i in range(config["train"]["num_models"]):
        print(f"\nModel {i + 1}\n")

        model = initialize_model(config)

        # Load model
        path_model = os.path.join(path_output_folder, f"model_{i}.pth")
        model.load(path_model)

        act_scr, pow_scr = get_model_scores(
            model,
            dl_test,
            config["data"]["power_scale"],
            means,
            thresholds,
            config["data"]["appliances"],
            config["data"]["threshold"]["min_off"],
            config["data"]["threshold"]["min_on"],
        )

        act_scores += act_scr
        pow_scores += pow_scr

        # Store individual scores
        act_dict = merge_dict_list(act_scr)
        pow_dict = merge_dict_list(pow_scr)

        scores = {"classification": act_dict, "regression": pow_dict}

        filename = f"scores_{i}.txt"

        if save_scores:
            store_scores(
                path_output_folder,
                config,
                scores,
                time_ellapsed,
                filename=filename,
            )

    # List scores

    scores = list_scores(
        config["data"]["appliances"],
        act_scores,
        pow_scores,
        config["train"]["num_models"],
    )

    # Store scores and plot
    if save_scores:
        store_scores(path_output_folder, config, scores, time_ellapsed)

    store_plots(
        path_output_folder,
        config,
        model,
        dl_test,
        means,
        thresholds
    )

    if save_predictions:
        store_real_data_and_predictions(
            path_output, config, model, dl_test, means, thresholds
        )


def main(
    path_data: str = "data/ukdale",
    path_output: str = "outputs",
    path_config: str = "better_nilm/config.toml",
    save_scores: bool = True,
    save_predictions: bool = True,
):
    """
    Trains several CONV models under the same conditions
    Stores scores and plots on results folder

    Parameters
    ----------
    path_data : str, optional
        Path to UK-DALE data
    path_output : str, optional
        Path to the results folder
    path_config : str, optional
        Path to the config toml file
    save_scores : bool, optional
        Store the model scores in txt files, by default True
    save_predictions : bool, optional
        Store the model predictions in txt files, by default True
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_conf_full(path_config)
    config = update_config(config)
    print("Done\n")

    assert os.path.isdir(path_data), "path_data must lead to folder:\n{}".format(
        path_data
    )
    path_h5 = path_data + ".h5"
    assert os.path.isfile(path_h5), "File not found:\n{}".format(path_h5)

    # Run main results
    print(f"{config['train']['name']}\n")

    test_many_models(
        path_h5,
        path_data,
        path_output,
        config,
        save_scores=save_scores,
        save_predictions=save_predictions,
    )

    print("PLOT RESULTS!")
    plot_scores_by_class_weight(config, path_output)


if __name__ == "__main__":
    typer.run(main)
