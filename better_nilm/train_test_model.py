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


def update_config(config: dict) -> dict:
    """Performs some corrections on the config dictionary"""
    if config["train"]["name"] == "GRUModel":
        border = int(
            (
                config["train"]["model"]["input_len"]
                - config["train"]["model"]["output_len"]
            )
            / 2
        )
        config["train"]["model"].update({"border": border})
    return config


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
    path_output = generate_path_output(path_output, config["train"]["name"])

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

        model = eval(config["train"]["name"])(**config["train"]["model"])

        # Train
        time_start = time.time()
        model.train_with_dataloader(
            dl_train,
            dl_valid,
            epochs=config["train"]["epochs"],
            patience=config["train"]["patience"],
        )
        time_ellapsed += time.time() - time_start

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
        config["train"]["num_models"],
    )

    time_ellapsed /= config["train"]["num_models"]

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

    run_many_models(path_h5, path_data, path_output, config)

    if config["plot"]["plot_scores"]:
        print("PLOT RESULTS!")
        nde_lim = config["plot"]["nde_lim"]
        f1_lim = config["plot"]["f1_lim"]
        for app in config["data"]["appliances"]:
            path_input = os.path.join(path_output, config["train"]["name"])
            # Folders related to the model we are working with
            model_name = (
                f"seq_{str(config['train']['model']['output_len'])}"
                f"_{config['data']['period']}"
                f"_{config['data']['threshold']['method']}"
            )
            # Store figures
            savefig = os.path.join(
                path_output,
                f"{config['train']['name']}"
                f"_{str(config['train']['model']['output_len'])}"
                f"_{config['data']['period']}"
                f"_{config['data']['threshold']['method']}_{app}.png",
            )
            plot_weights(
                path_input,
                app,
                model=model_name,
                figsize=config["plot"]["figsize"],
                savefig=savefig,
                nde_lim=nde_lim,
                f1_lim=f1_lim,
                dict_appliances=config["plot"]["appliances"],
            )
            print(f"Stored scores-weight plot in {savefig}")


if __name__ == "__main__":
    typer.run(main)
