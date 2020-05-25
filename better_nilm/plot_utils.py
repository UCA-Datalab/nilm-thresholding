import matplotlib.pyplot as plt
import numpy as np


def plot_real_vs_prediction(y_test, y_pred, idx=0,
                            sample_period=6, savefig=None,
                            threshold=None, y_total=None):
    """
    Plots the evolution of real and predicted appliance load values,
    assuming all are consecutive.

    Parameters
    ----------
    y_test : numpy.array
    y_pred : numpy.array
    idx : int, default=0
        Appliance index.
    sample_period : int, default=6
        Time between records, in seconds.
    savefig : str, default=None
        Path where the figure is stored.
    threshold : float, default=None
        If provided, draw the threshold line.
    y_total : numpy.array, default=None
        Total power load, from the main meter.

    """

    # Take just one appliance
    plt_test = y_test[:, :, idx].flatten().copy()
    plt_pred = y_pred[:, :, idx].flatten().copy()

    plt_x = np.arange(plt_test.shape[0]) * sample_period

    plt.figure(dpi=180)
    plt.plot(plt_x, plt_test)
    plt.plot(plt_x, plt_pred, alpha=.75)
        
    legend = ["Test", "Prediction"]
    
    if y_total is not None:
        plt_total = y_total[:, :, idx].flatten().copy()
        assert len(plt_total) == len(plt_test), "All arrays must have the " \
                                                "same length. "
        plt.plot(plt_x, plt_total, alpha=.75)
        legend += ["Total load"]

    if threshold is not None:
        plt.hlines(threshold, plt_x[0], plt_x[-1], colors='g',
                   linestyles='dashed')
    
    plt.ylabel("Load (w)")
    plt.xlabel("Time (s)")
    
    plt.legend(legend)
    if savefig is not None:
        plt.savefig(savefig)


def plot_load_and_state(load, state, idx=0,
                        sample_period=6, savefig=None):
    """
    Plots the evolution of load and state for the same appliance,
    assuming all values are consecutive.

    Parameters
    ----------
    load : numpy.array
    state : numpy.array
    idx : int, default=0
        Appliance index.
    sample_period : int, default=6
        Time between records, in seconds.
    savefig : str, default=None
        Path where the figure is stored.

    """
    # Take just one appliance
    plt_load = load[:, :, idx].flatten().copy()
    plt_state = state[:, :, idx].flatten().copy()

    plt_x = np.arange(plt_load.shape[0]) * sample_period

    fig, ax1 = plt.subplots(dpi=180)

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Load(w)', color=color)
    ax1.plot(plt_x, plt_load, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('State', color=color)
    ax2.plot(plt_x, plt_state, color=color, alpha=.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # Do the following. Otherwise the right y-label is slightly clipped
    fig.tight_layout()

    if savefig is not None:
        fig.savefig(savefig)
