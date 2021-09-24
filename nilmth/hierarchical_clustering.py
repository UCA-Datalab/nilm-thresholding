import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from nilmth.data.clustering import HierarchicalClustering
from nilmth.data.dataloader import DataLoader
from nilmth.data.threshold import Threshold
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


def plot_thresholds_on_distribution(
    ser: np.array,
    thresh: Iterable[float],
    ax: Optional[Axes] = None,
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
    ax : Axes, optional
        Axes where the graph is plotted
    app : str, optional
        Name of the appliance, by default ""
    bins : int, optional
        Histogram splits, by default 100
    """
    if ax is None:
        ax = plt.gca()
    y, x, _ = ax.hist(ser, bins=bins, range=(3, ser.max()))
    ax.set_title(app.capitalize().replace("_", " "))
    ax.set_xlabel("Power (watts)")
    ax.set_ylabel("Frequency")
    ax.set_title("Thresholds on power distribution (>=3 watts)")
    ax.grid()
    # Plot the thresholds
    for idx, t in enumerate(thresh):
        ax.axvline(t, color="r", linestyle="--")
        ax.text(t + 0.01 * x.max(), y.max(), idx + 1, rotation=0, color="r")


def plot_thresholds_on_series(
    ser: np.array, thresh: Iterable[float], size: int = 500, ax: Optional[Axes] = None
):
    """Plots the thresholds on a sample time series

    Parameters
    ----------
    ser : numpy.array
        Contains all the power values
    thresh : Iterable[float]
        Contains all the threshold values
    ax : Axes, optional
        Axes where the graph is plotted
    """
    if ax is None:
        ax = plt.gca()
    # Take sample of the series and plot it
    idx = np.argmax(ser)
    idx_min = int(max(0, idx - size / 2))
    idx_max = int(min(len(ser), idx + size / 2))
    power = ser[idx_min:idx_max]
    time = np.arange(0, len(power)) * 6
    ax.plot(time, power)
    # Plot the thresholds
    for idx, thresh in enumerate(thresh):
        ax.axhline(thresh, color="r", linestyle="--")
        ax.text(time.max(), thresh + power.max() * 0.01, idx + 1, rotation=0, color="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (watts)")
    ax.set_title("Thresholds on sample time series")
    ax.grid()


def plot_intrinsic_error(intr_error: Iterable[float], ax: Optional[Axes] = None):
    """Plots the intrinsic error depending on the number of splits

    Parameters
    ----------
    intr_error : Iterable[float]
        List of intrinsic error values
    ax : Axes, optional
        Axes where the graph is plotted
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(LIST_CLUSTER, intr_error, ".--")
    ax.set_ylabel("Intrinsic Error (NDE)")
    ax.set_xlabel("Number of status")
    ax.set_title("Intrinsic Error depending on splits")
    ax.grid()


def plot_error_reduction(intr_error: Iterable[float], ax: Optional[Axes] = None):
    """Plots the intrinsic error reduction depending on the number of splits

    Parameters
    ----------
    intr_error : Iterable[float]
        List of intrinsic error values
    ax : Axes, optional
        Axes where the graph is plotted
    """
    if ax is None:
        ax = plt.gca()
    rel_error = -100 * np.divide(np.diff(intr_error), intr_error[:-1])
    ax.plot(LIST_CLUSTER[1:], rel_error, ".--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Reduction of Intrinsic Error (%)")
    ax.set_xlabel("Number of status")
    ax.set_title("Error reduction with each subsequent split")
    ax.grid()


def plot_clustering_results(ser: np.array, method: str = "average"):
    """Plots the results of applying a certain clustering method
    on the given series

    Parameters
    ----------
    ser : np.array
        Contains all the power values
    method : str, optional
        Clustering method, by default "average"
    """
    # Clustering
    hie = HierarchicalClustering()
    hie.perform_clustering(ser, method=method)
    # Initialize the list of intrinsic error per number of clusters
    intr_error = [0] * len(LIST_CLUSTER)
    # Initialize threshold class
    th = Threshold()
    # Initialize the empty list of thresholds (sorted)
    thresh_sorted = []
    # Compute thresholds per number of clusters
    for idx, n_cluster in enumerate(LIST_CLUSTER):
        hie.compute_thresholds_and_centroids(n_cluster=n_cluster)
        # Update thresholds and centroids
        thresh = np.insert(np.expand_dims(hie.thresh, axis=0), 0, 0, axis=1)
        centroids = np.expand_dims(hie.centroids, axis=0)
        th.set_thresholds_and_centroids(thresh, centroids)
        # Create the dictionary of power series
        power = np.expand_dims(ser, axis=1)
        sta = th.power_to_status(power)
        recon = th.status_to_power(sta)
        dict_app = {"app": {"power": power, "power_pred": recon}}
        # Compute the scores
        dict_scores = regression_scores_dict(dict_app)
        intr_error[idx] = dict_scores["app"]["nde"]
        # Update the sorted list of thresholds
        thresh = list(set(hie.thresh) - set(thresh_sorted))
        thresh_sorted.append(thresh[0])
    # Initialize plots
    fig, axis = plt.subplots(2, 2, figsize=(12, 8))
    # Plots
    plot_thresholds_on_distribution(power, thresh_sorted, ax=axis[0, 0])
    plot_thresholds_on_series(power, thresh_sorted, ax=axis[0, 1])
    plot_intrinsic_error(intr_error, ax=axis[1, 0])
    plot_error_reduction(intr_error, ax=axis[1, 1])
    # Set the space between subplots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def main(
    limit: int = 20000,
    path_data: str = "data-prep",
    path_threshold: str = "threshold.toml",
    path_config: str = "nilmth/config.toml",
    path_output: str = "outputs/hieclust",
):
    """Performs the hierarchical clustering on the given list of appliances,
    testing different linkaged methods and number of splits. For each combination,
    outputs an image with several informative graphs.

    Parameters
    ----------
    limit : int, optional
        Number of data points to use, by default 20000
    path_data : str, optional
        Path to the preprocessed data folder, by default "data-prep"
    path_threshold : str, optional
        Path to the threshold configuration, by default "threshold.toml"
    path_config : str, optional
        Path to the config file, by default "nilmth/config.toml"
    path_output : str, optional
        Path to the outputs folder, by default "outputs/hieclust"
    """
    # Read config file
    config = load_config(path_config, "model")

    # Create output path
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # Loop through the list of appliances
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
        # Appliance name
        appliance = app.capitalize().replace("_", " ")
        # Loop through methods
        for method in LIST_LINKAGE:
            plot_clustering_results(ser, method=method)
            # Place title in figure
            plt.gcf().suptitle(f"{appliance}, Linkage: {method}")
            # Save and close the figure
            path_fig = os.path.join(path_output, f"{app}_{method}.png")
            plt.savefig(path_fig)
            plt.close()


if __name__ == "__main__":
    typer.run(main)
