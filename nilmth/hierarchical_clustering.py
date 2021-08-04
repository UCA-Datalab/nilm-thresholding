import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from nilmth.data.clustering import HierarchicalClustering
from nilmth.data.dataloader import DataLoader
from nilmth.utils.config import load_config
from nilmth.utils.scores import regression_scores_dict

LIST_APPLIANCES = ["dish_washer", "fridge", "washing_machine"]
LIST_CLUSTER = [2, 3, 4, 5, 6]
LIST_LINKAGE = [
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",  # Ward variance minimization algorithm
]


def plot_intrinsic_error(intr_error: Iterable[float], ax: Axes):
    """Plots the intrinsic error depending on the number of splits

    Parameters
    ----------
    intr_error : Iterable[float]
        List of intrinsic error values
    ax : Axes
        Axes where the graph is plotted
    """
    ax.plot(LIST_CLUSTER, intr_error, ".--")
    ax.set_ylabel("Intrinsic Error (NDE)")
    ax.set_xlabel("Number of status")
    ax.grid()


def plot_error_reduction(intr_error: Iterable[float], ax: Axes):
    """Plots the intrinsic error reduction depending on the number of splits

    Parameters
    ----------
    intr_error : Iterable[float]
        List of intrinsic error values
    ax : Axes
        Axes where the graph is plotted
    """
    rel_error = -100 * np.divide(np.diff(intr_error), intr_error[:-1])
    ax.plot(LIST_CLUSTER[1:], rel_error, ".--")
    ax.set_ylabel("Reduction of Intrinsic Error (%)")
    ax.set_xlabel("Number of status")
    ax.set_ylim(0, 100)
    ax.grid()


def plot_cluster_distribution(
    ser: np.array,
    thresh: Iterable[float],
    ax: Axes,
    app: str = "",
    bins: int = 100,
):
    """Plots the power distribution, and the lines splitting each cluster

    Parameters
    ----------
    ser : numpy.array
        Contains all the power values
    thresh : Iterable[float]
        Contains all the threshold values
    ax : Axes
        Axes where the graph is plotted
    app : str, optional
        Name of the appliance, by default ""
    bins : int, optional
        Histogram splits, by default 100
    """
    ax.hist(ser, bins=bins)
    ax.set_title(app.capitalize().replace("_", " "))
    ax.set_xlabel("Power (watts)")
    ax.set_ylabel("Frequency")
    ax.grid()
    [ax.axvline(t, color="r", linestyle="--") for t in thresh]


def main(
    limit: int = 20000,
    path_data: str = "data-prep",
    path_threshold: str = "threshold.toml",
    path_config: str = "nilmth/config.toml",
    path_output: str = "outputs/hieclust",
):
    # Read config file
    config = load_config(path_config, "model")

    # Create output path
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    for app in LIST_APPLIANCES:
        config["appliances"] = [app]

        # Prepare data loader with train data
        dl = DataLoader(
            path_data=path_data,
            subset="train",
            shuffle=False,
            path_threshold=path_threshold,
            **config,
        )

        # Take an appliance series
        ser = dl.get_appliance_series(app)[:limit]
        # Name
        appliance = app.capitalize().replace("_", " ")

        for method in LIST_LINKAGE:
            # Clustering
            hie = HierarchicalClustering()
            hie.perform_clustering(ser, method=method)
            # Initialize the list of intrinsic error
            # per number of clusters
            intr_error = [0] * len(LIST_CLUSTER)
            # Compute thresholds per number of clusters
            for idx, n_cluster in enumerate(LIST_CLUSTER):
                hie.compute_thresholds_and_centroids(n_cluster=n_cluster)
                # Update thresholds and centroids
                thresh = np.insert(np.expand_dims(hie.thresh, axis=0), 0, 0, axis=1)
                centroids = np.expand_dims(hie.centroids, axis=0)
                dl.threshold.set_thresholds_and_centroids(thresh, centroids)
                # Create the dictionary of power series
                power = np.expand_dims(ser, axis=1)
                sta = dl.dataset.power_to_status(power)
                recon = dl.dataset.status_to_power(sta)
                dict_app = {appliance: {"power": power, "power_pred": recon}}
                # Compute the scores
                dict_scores = regression_scores_dict(dict_app)
                intr_error[idx] = dict_scores[appliance]["nde"]
            # Initialize plots
            fig, axis = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"{appliance}, Linkage: {method}")
            # Plots
            plot_cluster_distribution(hie.x, hie.thresh, axis[0, 0])
            plot_intrinsic_error(intr_error, axis[1, 0])
            plot_error_reduction(intr_error, axis[1, 1])
            # Save and close the figure
            path_fig = os.path.join(path_output, f"{app}_{method}.png")
            fig.savefig(path_fig)
            plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
