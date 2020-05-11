import matplotlib.pyplot as plt
import numpy as np


def plot_real_vs_prediction(y_test, y_pred, idx=0, savefig=None):
    """
    Plots the evolution of real and predicted appliance load values,
    assuming all are consecutive.

    Parameters
    ----------
    y_test : numpy.array
    y_pred : numpy.array
    idx : int, default=0
        Appliance index
    savefig : str, default=None
        Path where the figure is stored

    """

    # Take just one appliance
    plt_test = y_test[:, :, idx].flatten().copy()
    plt_pred = y_pred[:, :, idx].flatten().copy()

    plt_x = np.arange(plt_test.shape[0])

    plt.figure(dpi=180)
    plt.plot(plt_x, plt_test)
    plt.plot(plt_x, plt_pred, alpha=.75)
    plt.legend(["Test", "Prediction"])
    if savefig is not None:
        plt.savefig(savefig)


def plot_load_and_state(load, state, idx=0, savefig=None):
    """
    Plots the evolution of load and state for the same appliance,
    assuming all values are consecutive.

    Parameters
    ----------
    load : numpy.array
    state : numpy.array
    idx : int, default=0
        Appliance index
    savefig : str, default=None
        Path where the figure is stored

    """
    # Take just one appliance
    plt_load = load[:, :, idx].flatten().copy()
    plt_state = state[:, :, idx].flatten().copy()

    plt_x = np.arange(plt_load.shape[0])

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (w)', color=color)
    ax1.plot(plt_x, plt_load, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('State', color=color)
    ax2.plot(plt_x, plt_state, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Do the following. Otherwise the right y-label is slightly clipped
    fig.tight_layout()

    if savefig is not None:
        fig.savefig(savefig)
