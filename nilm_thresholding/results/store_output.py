import collections
import os

import numpy as np
import pandas as pd

from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.plot import plot_informative_classification
from nilm_thresholding.utils.plot import plot_informative_regression


def list_scores(appliances, act_scores, pow_scores, num_models):
    """
    List scores in dictionary format.
    """
    scores = {"classification": {}, "regression": {}}

    for app in appliances:
        counter = collections.Counter()
        for sc in act_scores:
            counter.update(sc[app])
        scores["classification"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter).items()
        }

        counter = collections.Counter()
        for sc in pow_scores:
            counter.update(sc[app])
        scores["regression"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter).items()
        }

    return scores


def generate_path_output(path_output, model_name):
    """
    Generates results folder.
    """
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    path_output = os.path.join(path_output, model_name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def generate_folder_name(path_output: str, config: dict):
    """
    Generates specific folder inside results.
    """
    input_len = config["input_len"]
    period = config["period"]
    class_w = config["classification_w"]
    reg_w = config["regression_w"]
    threshold_method = config["threshold"]["method"]

    name = (
        f"seq_{str(input_len)}_{period}_{threshold_method}"
        f"_clas_{str(int(class_w * 100))}"
        f"_reg_{str(int(reg_w * 100))}"
    )
    path_output = os.path.join(path_output, name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def store_scores(
    path_output, config_model, scores, time_ellapsed, filename="scores.txt"
):
    # Load parameters
    class_w = config_model["classification_w"]
    reg_w = config_model["regression_w"]
    train_size = config_model["train_size"]
    valid_size = config_model["valid_size"]
    num_models = config_model["num_models"]
    batch_size = config_model["batch_size"]
    learning_rate = config_model["learning_rate"]
    dropout = config_model["dropout"]
    epochs = config_model["epochs"]
    patience = config_model["patience"]

    path_scores = os.path.join(path_output, filename)

    with open(path_scores, "w") as text_file:
        text_file.write(
            f"Train size: {train_size}\n"
            f"Validation size: {valid_size}\n"
            f"Number of models: {num_models}\n"
            f"Batch size: {batch_size}\n"
            f"Learning rate: {learning_rate}\n"
            f"Dropout: {dropout}\n"
            f"Epochs: {epochs}\n"
            f"Patience: {patience}\n"
            f"Time per model (seconds): {time_ellapsed}\n"
            f"=============================================\n"
        )
        for key, dic1 in scores.items():

            # Skip scores if weight is zero
            if (class_w == 0) and (key.startswith("class")):
                continue
            if (reg_w == 0) and (key.startswith("reg")):
                continue

            text_file.write(f"{key}\n------------------------------------------\n")
            for app, dic2 in dic1.items():
                text_file.write(f"{app} \n")
                for name, value in dic2.items():
                    text_file.write(f"{name}: {value}\n")
                text_file.write("----------------------------------------------\n")
            text_file.write("==================================================\n")


def store_plots(path_output, config, model, dl_test, means, thresholds):

    # Ensure appliances is a list
    appliances = to_list(config["data"]["appliances"])

    # Compute period of x axis
    if config["data"]["period"].endswith("min"):
        period_x = int(config["data"]["period"].replace("min", ""))
    elif config["data"]["period"].endswith("s"):
        period_x = float(config["data"]["period"].replace("s", "")) / 60

    # Model values

    x, p_true, s_true, p_hat, s_hat = model.predict(dl_test)

    p_true, p_hat, s_hat, sp_hat, ps_hat = model.process_outputs(
        p_true, p_hat, s_hat, dl_test
    )

    thresh_color = config["plot"]["thresh_color"].get(
        config["data"]["threshold"]["method"], "grey"
    )

    for idx, app in enumerate(appliances):

        # Plot a certain number of sequences per appliance
        idx_start = 0
        num_plots = 0
        while (num_plots < config["plot"]["num_plots"]) and (
            (idx_start + config_model["name"]["output_len"]) < p_true.shape[0]
        ):
            idx_end = idx_start + config_model["name"]["output_len"]
            p_t = p_true[idx_start:idx_end, idx]
            if p_t.sum() > 0:
                s_t = s_true[idx_start:idx_end, idx]
                s_h = s_hat[idx_start:idx_end, idx]
                p_h = p_hat[idx_start:idx_end, idx]
                num_plots += 1

                # Add aggregate load. Try to de-normalize it
                p_agg = np.multiply(x[idx_start:idx_end], config["data"]["power_scale"])
                p_agg -= p_agg.min()
                # It may need further denormalization if one of its values
                # is lower than the appliance load
                factor = (p_agg - p_t).min()
                if factor < 0:
                    p_agg -= factor

                idx_start += config_model["name"]["output_len"]
            else:
                idx_start += config_model["name"]["output_len"]
                continue
            # Skip plots if weight is zero
            if config_model["name"]["classification_w"] > 0:
                savefig = os.path.join(
                    path_output, f"{app}_classification_{num_plots}.png"
                )
                plot_informative_classification(
                    s_t,
                    s_h,
                    p_agg,
                    records=config_model["name"]["output_len"],
                    period=period_x,
                    pw_max=p_agg.max(),
                    dpi=180,
                    thresh_color=thresh_color,
                    savefig=savefig,
                    title=app,
                )
            if config_model["name"]["regression_w"] > 0:
                savefig = os.path.join(path_output, f"{app}_regression_{num_plots}.png")
                plot_informative_regression(
                    p_t,
                    p_h,
                    p_agg,
                    records=config_model["model"]["output_len"],
                    scale=1.0,
                    period=period_x,
                    dpi=180,
                    savefig=savefig,
                    title=app,
                )


def store_real_data_and_predictions(
    path_output, config, model, dl_test, means, thresholds
):

    # Ensure appliances is a list
    appliances = to_list(config["data"]["appliances"])

    # Model values

    x, p_true, s_true, p_hat, s_hat = model.predict(dl_test)

    p_true, p_hat, s_hat, sp_hat, ps_hat = model.process_outputs(
        p_true, p_hat, s_hat, dl_test
    )

    for idx, app in enumerate(appliances):
        # Store results
        df = pd.DataFrame(
            {
                "x": x,
                "y_true": p_true[:, idx],
                "y_hat": p_hat[:, idx],
                "s_true": s_true[:, idx],
                "s_hat": s_hat[:, idx],
            }
        )
        save_csv = os.path.join(path_output, f"{app}_data.csv")
        df.to_csv(save_csv)
