import os
import re
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("white")


def list_dir_sorted(path_input, model):
    files = []
    for i in os.listdir(path_input):
        if os.path.isdir(os.path.join(path_input, i)) and model in i:
            files.append(
                (
                    os.path.join(path_input, i),
                    float(re.search("clas_(.*)_reg", i).group(1)),
                )
            )
    files.sort(key=lambda tup: tup[1])
    return files


def list_scores(path_dir):
    list_scores_txt = []
    for i in os.listdir(path_dir):
        if os.path.isfile(os.path.join(path_dir, i)) and "scores_" in i:
            list_scores_txt.append(os.path.join(path_dir, i))
    return list_scores_txt


def extract_scores_from_file(path_file):
    with open(path_file, "r") as search:
        next_line_is = None
        approach = None
        appliance = None
        scores = {}
        for line in search:
            line = line.rstrip()
            if line.startswith("==="):
                next_line_is = "approach"
                approach = None
            elif line.startswith("---"):
                next_line_is = "appliance"
                appliance = None
            elif next_line_is == "approach":
                approach = line
                scores[approach] = {}
                next_line_is = None
            elif next_line_is == "appliance":
                appliance = line
                scores[approach][appliance] = {}
                next_line_is = "score"
            elif next_line_is == "score":
                key = line.split(": ", 1)[0]
                value = float(line.split(": ", 1)[1])
                scores[approach][appliance][key] = value
    return scores


def get_f1_nde_from_scores(list_scores_txt):
    list_f1 = []
    list_nde = []
    for path_file in list_scores_txt:
        dict_scores = extract_scores_from_file(path_file)
        # Include F1 from classification
        # If missing, get F1 from regression
        if "classification" in dict_scores.keys():
            for app, dic in dict_scores["classification"].items():
                list_f1.append((app, dic["f1"]))
        elif "regression" in dict_scores.keys():
            for app, dic in dict_scores["regression"].items():
                list_f1.append((app, dic["f1"]))
        # Include NDE from regression
        # If missing, get NDE from classification
        if "regression" in dict_scores.keys():
            for app, dic in dict_scores["regression"].items():
                list_nde.append((app, dic["nde"]))
        elif "classification" in dict_scores.keys():
            for app, dic in dict_scores["classification"].items():
                list_nde.append((app, dic["nde"]))
    list_f1.sort(key=lambda tup: tup[0])
    dict_f1 = dict(
        [(k, list(list(zip(*g))[1])) for k, g in groupby(list_f1, itemgetter(0))]
    )
    list_nde.sort(key=lambda tup: tup[0])
    dict_nde = dict(
        [(k, list(list(zip(*g))[1])) for k, g in groupby(list_nde, itemgetter(0))]
    )
    return dict_f1, dict_nde


def get_arrays(list_values):
    w = np.zeros(len(list_values))
    mean = np.zeros(len(list_values))
    std = np.zeros(len(list_values))
    for i, (weight, values) in enumerate(list_values):
        w[i] = weight
        mean[i] = np.mean(values)
        std[i] = np.std(values)
    return w, mean, std


def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1 :] = ret[n - 1 :] / n
    return ret


def subplot_f1(ax, w, y, std, ignore_extreme, f1_lim=(0.5, 1)):
    color = "tab:red"
    ax.set_xlabel("Classification weight")
    ax.set_ylabel("F1", color=color)
    up = y + std
    down = y - std
    if ignore_extreme or w.min() != 0:
        ax.plot(w[1:], y[1:], color=color)
        ax.fill_between(w[1:], down[1:], up[1:], color=color, alpha=0.2)
    else:
        ax.plot(w[1:], y[1:], color=color)
        ax.fill_between(w[1:], down[1:], up[1:], color=color, alpha=0.2)
        ax.errorbar(w[0], y[0], std[0], color=color, linestyle="None", marker=".")
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(axis="y")
    ax.grid(axis="x")
    if f1_lim is not None:
        ax.set_ylim(f1_lim)
    return ax


def subplot_nde(ax, w, y, std, ignore_extreme, nde_lim={}):
    color = "tab:blue"
    ax.set_ylabel("NDE", color=color)  # we already handled the x-label with ax1
    up = y + std
    down = y - std
    if ignore_extreme or w.max() != 1:
        ax.plot(w[:-1], y[:-1], color=color)
        ax.fill_between(w[:-1], down[:-1], up[:-1], color=color, alpha=0.2)
    else:
        ax.plot(w[:-1], y[:-1], color=color)
        ax.fill_between(w[:-1], down[:-1], up[:-1], color=color, alpha=0.2)
        ax.errorbar(w[-1], y[-1], std[-1], color=color, linestyle="None", marker=".")
    ax.tick_params(axis="y", labelcolor=color)
    if nde_lim is not None:
        ax.set_ylim(nde_lim)
    return ax


def plot_arrays(
    w_f1,
    f1,
    f1_std,
    w_nde,
    nde,
    nde_std,
    app,
    model,
    nde_lim=(0, 1),
    f1_lim=(0, 1),
    dict_appliances={},
    movavg=1,
    ignore_extreme=True,
    figsize=(6, 4),
    savefig=None,
):
    app = dict_appliances.get(app, app)
    if movavg > 1:
        f1 = moving_average(f1, n=movavg)
        nde = moving_average(nde, n=movavg)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1 = subplot_f1(ax1, w_f1, f1, f1_std, ignore_extreme, f1_lim=f1_lim)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2 = subplot_nde(ax2, w_nde, nde, nde_std, ignore_extreme, nde_lim=nde_lim)

    ax1.set_title(model + " " + app)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def _plot_weights(
    path_input: str,
    app: str,
    nde_lim=(0, 1),
    f1_lim=(0, 1),
    dict_appliances={},
    model: str = "seq_480_1min",
    movavg: int = 1,
    ignore_extreme: bool = False,
    figsize=(8, 4),
    savefig=None,
):
    assert os.path.isdir(path_input)

    # List files and sort by class weight
    files = list_dir_sorted(path_input, model)

    list_f1 = []
    list_nde = []

    for path_dir, clas_w in files:
        list_scores_txt = list_scores(path_dir)
        dict_f1, dict_nde = get_f1_nde_from_scores(list_scores_txt)
        if app in dict_f1.keys():
            list_f1.append((clas_w / 100, dict_f1[app]))
        if app in dict_nde.keys():
            list_nde.append((clas_w / 100, dict_nde[app]))

    # Build arrays
    w_f1, f1, f1_std = get_arrays(list_f1)
    w_nde, nde, nde_std = get_arrays(list_nde)

    # Plot arrays
    model = path_input.rsplit("/", 1)[1]
    model = model[:-5]
    plot_arrays(
        w_f1,
        f1,
        f1_std,
        w_nde,
        nde,
        nde_std,
        app,
        model,
        nde_lim=nde_lim,
        f1_lim=f1_lim,
        dict_appliances=dict_appliances,
        movavg=movavg,
        ignore_extreme=ignore_extreme,
        figsize=figsize,
        savefig=savefig,
    )


def plot_scores_by_class_weight(config, path_output):
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
        _plot_weights(
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
