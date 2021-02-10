import os
import time
from collections import defaultdict

import typer

from better_nilm.data.ukdale import load_dataloaders
from better_nilm.results.store_output import (
    generate_path_output,
    get_model_scores,
    list_scores,
    store_plots,
    store_scores,
)
from better_nilm.results.plot_output import plot_weights
from better_nilm.utils.conf import load_conf_full

from better_nilm.model.architecture.conv import ConvModel
from better_nilm.model.architecture.gru import GRUModel


def _merge_dict_list(dict_list):
    d = defaultdict(dict)
    for l in dict_list:
        for elem in l:
            d[elem].update(l[elem])

    return d


def run_many_models(path_h5, path_data, path_output, config: dict):
    """
    Runs several models with the same conditions.
    Stores plots and the average scores of those models.
    """

    # Set output path
    path_output = generate_path_output(path_output, config["model"]["name"])

    # Load data

    params = load_dataloaders(path_h5, path_data, config)

    dl_train, dl_valid, dl_test, kmeans = params
    thresholds, means = kmeans

    # Training

    act_scores = []
    pow_scores = []
    time_ellapsed = 0

    for i in range(config["model"]["num_models"]):
        print(f"\nModel {i + 1}\n")

        model = eval(config["model"]["name"])(**config["model"]["params"])

        # Train
        time_start = time.time()
        model.train_with_dataloader(
            dl_train,
            dl_valid,
            epochs=config["model"]["epochs"],
            patience=config["model"]["patience"],
        )
        time_ellapsed += time.time() - time_start

        act_scr, pow_scr = get_model_scores(
            model,
            dl_test,
            config["model"]["power_scale"],
            means,
            thresholds,
            config["data"]["appliances"],
            config["model"]["min_off"],
            config["model"]["min_on"],
        )

        act_scores += act_scr
        pow_scores += pow_scr

        # Store individual scores
        act_dict = _merge_dict_list(act_scr)
        pow_dict = _merge_dict_list(pow_scr)

        scores = {"classification": act_dict, "regression": pow_dict}

        filename = "scores_{}.txt".format(i)

        store_scores(
            path_output,
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
        config["model"]["num_models"],
    )

    time_ellapsed /= config["model"]["num_models"]

    # Store scores and plot

    store_scores(
        path_output,
        config,
        scores,
        time_ellapsed,
    )

    store_plots(
        path_output,
        config,
        model,
        dl_test,
        means,
        thresholds,
    )


def main(
    path_data: str = "data/ukdale",
    path_output: str = "outputs",
    path_config: str = "better_nilm/config.toml",
    model_name: str = "ConvModel",
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
    model_name : str, optional
        Model to train, by default "ConvModel"
        Other option is "GRUModel"
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_conf_full(path_config)
    config["model"].update({"name": model_name})
    print("Done\n")

    assert os.path.isdir(path_data), "path_data must lead to folder:\n{}".format(
        path_data
    )
    path_h5 = path_data + ".h5"
    assert os.path.isfile(path_h5), "File not found:\n{}".format(path_h5)

    if not os.path.exists(path_output):
        print(f"Output path not found. Creating: {path_output}")
        os.mkdir(path_output)
        print("Done\n")

    # Run main results
    print(f"{model_name}\n")

    run_many_models(path_h5, path_data, path_output, config)

    if config["plot"]["plot_scores"]:
        print("PLOT RESULTS!")
        if config["plot"]["plot_scores_lim"]:
            dict_mae_lim = config["plot"]["mae_lim"]
            f1_lim = config["plot"]["f1_lim"]
        else:
            dict_mae_lim = {}
            f1_lim = None
        for app in config["data"]["appliances"]:
            path_input = os.path.join(path_output, config["model"]["name"])
            savefig = os.path.join(path_output, "Conv_" + app + ".png")
            plot_weights(
                path_input,
                app,
                figsize=(4, 3),
                savefig=savefig,
                dict_mae_lim=dict_mae_lim,
                f1_lim=f1_lim,
                dict_appliances=config["plot"]["appliances"],
            )


if __name__ == "__main__":
    typer.run(main)
