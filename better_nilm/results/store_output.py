import collections
import os

import numpy as np
import pandas as pd

from better_nilm.utils.format_list import to_list
from better_nilm.model.preprocessing import get_status
from better_nilm.model.preprocessing import get_status_by_duration
from better_nilm.utils.scores import classification_scores_dict
from better_nilm.utils.scores import regression_scores_dict
from better_nilm.utils.plot import plot_informative_classification
from better_nilm.utils.plot import plot_informative_regression


def process_model_outputs(
    p_true, p_hat, s_hat, power_scale, means, thresholds, min_off, min_on
):
    # Denormalize power values
    p_true = np.multiply(p_true, power_scale)
    p_hat = np.multiply(p_hat, power_scale)
    p_hat[p_hat < 0.0] = 0.0

    # Get status
    if (min_on is None) or (min_off is None):
        s_hat[s_hat >= 0.5] = 1
        s_hat[s_hat < 0.5] = 0
    else:
        thresh = [0.5] * len(min_on)
        s_hat = get_status_by_duration(s_hat, thresh, min_off, min_on)

    # Get power values from status
    sp_hat = np.multiply(np.ones(s_hat.shape), means[:, 0])
    sp_on = np.multiply(np.ones(s_hat.shape), means[:, 1])
    sp_hat[s_hat == 1] = sp_on[s_hat == 1]

    # Get status from power values
    ps_hat = get_status(p_hat, thresholds)

    return p_true, p_hat, s_hat, sp_hat, ps_hat


def get_model_scores(
    model, dl_test, power_scale, means, thresholds, appliances, min_off, min_on
):
    """
    Trains and test a model. Returns its activation and power scores.
    """

    # Test
    x_true, p_true, s_true, p_hat, s_hat = model.predict_loader(dl_test)

    p_true, p_hat, s_hat, sp_hat, ps_hat = process_model_outputs(
        p_true, p_hat, s_hat, power_scale, means, thresholds, min_off, min_on
    )

    # classification scores

    class_scores = classification_scores_dict(s_hat, s_true, appliances)
    reg_scores = regression_scores_dict(sp_hat, p_true, appliances)
    act_scores = [class_scores, reg_scores]

    print("classification scores")
    print(class_scores)
    print(reg_scores)

    # regression scores

    class_scores = classification_scores_dict(ps_hat, s_true, appliances)
    reg_scores = regression_scores_dict(p_hat, p_true, appliances)
    pow_scores = [class_scores, reg_scores]

    print("regression scores")
    print(class_scores)
    print(reg_scores)

    return act_scores, pow_scores


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


def generate_path_output(path_main, model_name):
    """
    Generates results folder.
    """
    path_output = os.path.join(path_main, "results")
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    path_output = os.path.join(path_output, model_name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def generate_folder_name(
    path_output, output_len, period, class_w, reg_w, threshold_method
):
    """
    Generates specific folder inside results.
    """
    name = (
        f"seq_{str(output_len)}_{period}_clas_{str(int(class_w * 100))}"
        f"_reg_{str(int(reg_w * 100))}_{threshold_method}"
    )
    path_output = os.path.join(path_output, name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def store_scores(path_output, config, scores, time_ellapsed, filename="scores.txt"):
    # Load parameters
    output_len = config["model"]["params"]["output_len"]
    period = config["data"]["period"]
    class_w = config["model"]["params"]["classification_w"]
    reg_w = config["model"]["params"]["regression_w"]
    threshold_method = config["data"]["threshold"]["method"]
    train_size = config["model"]["train_size"]
    valid_size = config["model"]["valid_size"]
    num_models = config["model"]["num_models"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["params"]["learning_rate"]
    dropout = config["model"]["params"]["dropout"]
    epochs = config["model"]["epochs"]
    patience = config["model"]["patience"]

    path_output = generate_folder_name(
        path_output, output_len, period, class_w, reg_w, threshold_method
    )

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
            # if (class_w == 0) and (key.startswith('class')):
            #    continue
            # if (reg_w == 0) and (key.startswith('reg')):
            #    continue

            text_file.write(f"{key}\n------------------------------------------\n")
            for app, dic2 in dic1.items():
                text_file.write(f"{app} \n")
                for name, value in dic2.items():
                    text_file.write(f"{name}: {value}\n")
                text_file.write("----------------------------------------------\n")
            text_file.write("==================================================\n")


def store_plots(path_output, config, model, dl_test, means, thresholds):
    path_output = generate_folder_name(
        path_output,
        config["model"]["output_len"],
        config["data"]["period"],
        config["model"]["classification_w"],
        config["model"]["regression_w"],
        config["model"]["threshold_method"],
    )

    # Ensure appliances is a list
    appliances = to_list(config["data"]["appliances"])

    # Compute period of x axis
    if config["data"]["period"].endswith("min"):
        period_x = int(config["data"]["period"].replace("min", ""))
    elif config["data"]["period"].endswith("s"):
        period_x = float(config["data"]["period"].replace("s", "")) / 60

    # Model values

    x, p_true, s_true, p_hat, s_hat = model.predict_loader(dl_test)

    p_true, p_hat, s_hat, sp_hat, ps_hat = process_model_outputs(
        p_true,
        p_hat,
        s_hat,
        config["data"]["power_scale"],
        means,
        thresholds,
        config["data"]["threshold"]["min_off"],
        config["data"]["threshold"]["min_on"],
    )

    thresh_color = config["plot"]["thresh_color"].get(
        config["model"]["threshold_method"], "grey"
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

        # Plot a certain number of sequences per appliance
        idx_start = 0
        num_plots = 0
        while (num_plots < 10) and (
            (idx_start + config["model"]["output_len"]) < p_true.shape[0]
        ):
            idx_end = idx_start + config["model"]["output_len"]
            p_t = p_true[idx_start:idx_end, idx]
            if p_t.sum() > 0:
                s_t = s_true[idx_start:idx_end, idx]
                sp_h = sp_hat[idx_start:idx_end, idx]
                s_h = s_hat[idx_start:idx_end, idx]
                p_h = p_hat[idx_start:idx_end, idx]
                ps_h = ps_hat[idx_start:idx_end, idx]
                num_plots += 1

                # Add aggregate load. Try to de-normalize it
                p_agg = np.multiply(x[idx_start:idx_end], config["data"]["power_scale"])
                p_agg -= p_agg.min()
                # It may need further denormalization if one of its values
                # is lower than the appliance load
                factor = (p_agg - p_t).min()
                if factor < 0:
                    p_agg -= factor

                idx_start += config["model"]["output_len"]
            else:
                idx_start += config["model"]["output_len"]
                continue
            # Skip plots if weight is zero
            if config["model"]["classification_w"] > 0:
                savefig = os.path.join(
                    path_output, f"{app}_classification_{num_plots}.png"
                )
                plot_informative_classification(
                    s_t,
                    s_h,
                    p_agg,
                    records=config["model"]["output_len"],
                    period=period_x,
                    pw_max=p_agg.max(),
                    dpi=180,
                    thresh_color=thresh_color,
                    savefig=savefig,
                    title=app,
                )
            if config["model"]["regression_w"] > 0:
                savefig = os.path.join(path_output, f"{app}_regression_{num_plots}.png")
                plot_informative_regression(
                    p_t,
                    p_h,
                    p_agg,
                    records=config["model"]["output_len"],
                    scale=1.0,
                    period=period_x,
                    dpi=180,
                    savefig=savefig,
                    title=app,
                )
