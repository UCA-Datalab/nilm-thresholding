import matplotlib.pyplot as plt
import numpy as np


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
    for app in list_app:
        name = app.capitalize().replace("_", " ")
        y = dict_pred[app]["power"][idx_start:idx_end]
        ax.plot(x, y, label=name)
    ax.legend()
    ax.set_ylabel("Power (watts)")
    ax.set_xlabel("Time (minutes)")
    ax.grid()
    if savefig is None:
        return fig, ax
    else:
        plt.savefig(savefig)
        plt.close()
