import matplotlib.pyplot as plt
import numpy as np


def _flatten_series(y_a, y_b, idx, num_series):
    # Take just one appliance
    if num_series is None:
        plt_a = y_a[:, :, idx].flatten().copy()
        plt_b = y_b[:, :, idx].flatten().copy()
    else:
        # Take a limited number of series
        # Ensure there is at least one activation in the series
        plt_a = np.array([0])
        i = 0
        j = num_series
        while plt_a.sum() == 0:
            plt_a = y_a[i:j, :, idx].flatten().copy()
            plt_b = y_b[i:j, :, idx].flatten().copy()
            i += num_series
            j += num_series

    return plt_a, plt_b


def plot_real_vs_prediction(y_test, y_pred, idx=0,
                            sample_period=6, num_series=None, savefig=None,
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
    num_series : int, default=None
        If given, limit the number of plotted sequences.
    savefig : str, default=None
        Path where the figure is stored.
    threshold : float, default=None
        If provided, draw the threshold line.
    y_total : numpy.array, default=None
        Total power load, from the main meter.

    """
    # Take just one appliance and flatten the series
    plt_test, plt_pred = _flatten_series(y_test, y_pred, idx, num_series)

    plt_x = np.arange(plt_test.shape[0]) * sample_period
    # Set time units
    if plt_x.max() >= 1e3:
        plt_x = plt_x / 60
        time_units = "min"
    else:
        time_units = "s"

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

    # If no value has gone higher than 1, we assume we are working with
    # binary states. Otherwise, it is load
    if plt_test.max() > 1:
        plt.ylabel("Load (w)")
    else:
        plt.ylabel("State")
    plt.xlabel(f"Time ({time_units})")
    
    plt.legend(legend)
    if savefig is not None:
        plt.savefig(savefig)


def plot_load_and_state(load, state, idx=0,
                        sample_period=6, num_series=None, savefig=None):
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
    num_series : int, default=None
        If given, limit the number of plotted sequences.
    savefig : str, default=None
        Path where the figure is stored.

    """
    # Take just one appliance and flatten the series
    plt_state, plt_load = _flatten_series(state, load, idx, num_series)

    plt_x = np.arange(plt_load.shape[0]) * sample_period
    # Set time units
    if plt_x.max() >= 1e3:
        plt_x = plt_x / 60
        time_units = "min"
    else:
        time_units = "s"

    fig, ax1 = plt.subplots(dpi=180)

    color = 'tab:red'
    ax1.set_xlabel(f"Time ({time_units})")
    ax1.set_ylabel('Load (w)', color=color)
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
