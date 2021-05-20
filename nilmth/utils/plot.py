import numpy as np
from matplotlib import pyplot as plt


def plot_real_data(
    dict_pred: dict,
    figsize: tuple = (10, 6),
    idx_start: int = 0,
    idx_end: int = 480,
    freq_min: float = 1,
    savefig: str = None,
):
    """Plot the aggregated power load and each appliances' consumption"""
    fig, ax = plt.subplots(figsize=figsize)
    # Plot aggregate power
    y = dict_pred["aggregated"][idx_start:idx_end]
    x = np.arange(0, len(y)) * freq_min
    ax.plot(x, y, label="Aggregate")
    # List appliances
    list_app = list(dict_pred.keys())
    list_app.remove("aggregated")
    # Plot each appliance
    for app in list_app:
        name = app.capitalize().replace("_", " ")
        y = dict_pred[app]["power"][idx_start:idx_end]
        ax.plot(x, y, label=name)
    # Add plot information
    ax.legend()
    ax.set_ylabel("Power (watts)")
    ax.set_xlabel("Time (minutes)")
    ax.grid()
    if savefig is None:
        return fig, ax
    else:
        plt.savefig(savefig)
        plt.close()


def plot_real_vs_prediction(
    dict_pred: dict,
    appliance: str,
    figsize: tuple = (10, 6),
    idx_start: int = 0,
    idx_end: int = 480,
    freq_min: float = 1,
    savefig: str = None,
):
    """Plot the real and predicted power consumption of an appliance"""
    fig, ax = plt.subplots(figsize=figsize)
    # Plot real power
    y = dict_pred[appliance]["power"][idx_start:idx_end]
    x = np.arange(0, len(y)) * freq_min
    ax.plot(x, y, label="Real")
    # Plot prediction
    y = dict_pred[appliance]["power_pred"][idx_start:idx_end]
    ax.plot(x, y, label="Prediction")
    # Add plot information
    name = appliance.capitalize().replace("_", " ")
    ax.set_title(name)
    ax.legend()
    ax.set_ylabel("Power (watts)")
    ax.set_xlabel("Time (minutes)")
    ax.grid()
    if savefig is None:
        return fig, ax
    else:
        plt.savefig(savefig)
        plt.close()


def plot_power_distribution(
    ser: np.array, app: str, bins: int = 20, figsize: tuple = (3, 3)
):
    """Plot the histogram of power distribution

    Parameters
    ----------
    ser : numpy.array
        Contains all the power values
    app : label
        Name of the appliance
    bins : int, optional
        Histogram splits, by default 20
    figsize : tuple, optional
        Figure size, by default (3, 3)

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes._subplots.AxesSubplot

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(ser, bins=bins)
    ax.set_title(app.capitalize().replace("_", " "))
    ax.set_xlabel("Power (watts)")
    ax.set_ylabel("Frequency")
    ax.grid()
    return fig, ax
