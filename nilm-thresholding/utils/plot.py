import matplotlib.pyplot as plt
import numpy as np


def _flatten_series(y_a, y_b, idx, num_series):
    """
    Flat tensor of 3-dims into 2-d matrix
    """
    # Take just one appliance
    if num_series is None:
        if len(y_a.shape) == 3:
            plt_a = y_a[:, :, idx].flatten().copy()
            plt_b = y_b[:, :, idx].flatten().copy()
        else:
            plt_a = y_a[:, idx].flatten().copy()
            plt_b = y_b[:, idx].flatten().copy()
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
                            sample_period=6, num_series=None,
                            dpi=180, savefig=None,
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
    dpi : int, default=180
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

    plt.figure(dpi=dpi)
    plt.plot(plt_x, plt_test)
    plt.plot(plt_x, plt_pred, alpha=.75)

    legend = ["True", "Prediction"]

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


def plot_informative_sample(p_true, s_true, p_hat, s_hat, p_agg=None,
                            records=480, app_idx=0, scale=1.,
                            period=1., dpi=100, savefig=None,
                            thresh_color='grey'):
    """

    Parameters
    ----------
    p_true : numpy.array
        shape = (total_records, num_appliances)
    s_true : numpy.array
        shape = (total_records, num_appliances)
    p_hat : numpy.array
        shape = (total_records, num_appliances)
    s_hat : numpy.array
        shape = (total_records, num_appliances)
    p_agg : numpy.array
        shape = (total_records)
        Aggregate power
    records : int, default=480
        Number of records to plot
    app_idx : int, default = 0
        Appliance index
    scale : float, default=1.
        Value to multiply the load
    period : float, default=1.
        Sample period, in minutes
    dpi : int, default=100
        Dots per inch, image quality
    savefig : str, default=None
        Path where the figure is stored.
    thresh_color : str, default='grey'
        Color of the threshold bars, when appliance state is ON
    """
    # We dont want to modify the originals
    p_true = p_true.copy()
    s_true = s_true.copy()
    p_hat = p_hat.copy()
    s_hat = s_hat.copy()

    # Define time
    t = np.array([i for i in range(records)])
    t = np.multiply(t, period)

    # Take appliance power and scale it
    pw = p_true[:, app_idx] * scale
    pw_pred = p_hat[:, app_idx] * scale

    # Take power in given interval and get the maximum value
    pw_max = max(pw.max(), pw_pred.max())

    # Take status in given interval, only ON activations
    s_hat = s_hat[:, app_idx]
    s_true = s_true[:, app_idx]

    mask_on = s_hat == 1
    s_hat = s_hat[mask_on].copy()
    t_on = t[mask_on].copy()

    # Distinguish between correct and incorrect guesses
    # Scale status to power, to see it properly
    s_scaled = s_hat.copy() + .01
    s_scaled = np.multiply(s_scaled, pw_max)

    # Show all ON guesses in the same color
    s_hat = s_hat.copy() + .05
    s_hat = np.multiply(s_hat, pw_max)

    # Plot the figure
    plt.figure(dpi=dpi)

    plt.bar(t, np.multiply(s_true, pw_max * 2),
            color=thresh_color, alpha=.2, width=1,
            label='Appliance status')

    if p_agg is not None:
        plt.plot(t, p_agg, '--', color='grey',
                 label='Aggregate power')

    plt.plot(t, pw, color='black', label='Appliance power')
    plt.plot(t, pw_pred, color='blue', label='Predicted power')

    plt.scatter(t_on, s_hat, color='blue', s=.8,
                label='Predicted ON activation')

    plt.legend()

    plt.ylim([0, pw_max * 1.1])
    plt.ylabel('Power (watts)')
    plt.xlabel('Time (minutes)')

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()


def plot_informative_classification(s_true, s_hat, p_agg,
                                    records=480, pw_max=1.,
                                    period=1., dpi=100, savefig=None,
                                    thresh_color='grey', title=None):
    """
    Plots the following:
    - Predicted appliance state (s_hat) as dots
    - Real appliance state (s_true) as bars when ON
    - Real aggregated load (p_agg) as a grey line, in watts
    """
    # We dont want to modify the originals
    s_true = s_true.copy()
    s_hat = s_hat.copy()
    p_agg = p_agg.copy()

    # Define time
    t = np.array([i for i in range(records)])
    t = np.multiply(t, period)

    # Get ON activations
    mask_on = s_hat == 1
    s_hat = s_hat[mask_on].copy() * pw_max * .75
    t_on = t[mask_on].copy()

    # Plot the figure
    plt.figure(dpi=dpi)

    plt.fill_between(t, p_agg, color='grey', alpha=.4, label='Aggregate power')
    plt.bar(t, np.multiply(s_true, pw_max * 2),
            color=thresh_color, alpha=.2, width=1, label='Appliance status')
    plt.scatter(t_on, s_hat, color=thresh_color, s=.6,
                label='Predicted ON activation')

    plt.legend()

    plt.ylim([0, pw_max])
    plt.ylabel('Power (watts)')
    plt.xlabel('Time (minutes)')

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()


def plot_informative_regression(p_true, p_hat, p_agg,
                                records=480, scale=1.,
                                period=1., dpi=100, savefig=None,
                                color='purple', title=None):
    """
    Plots the following:
    - Predicted appliance power (p_hat) as a line, in watts
    - Real appliance power (p_true) as a line, in watts
    - Real aggregated load (p_agg) as a grey line, in watts
    """
    # We dont want to modify the originals
    p_true = p_true.copy()
    p_hat = p_hat.copy()
    p_agg = p_agg.copy()

    # Define time
    t = np.array([i for i in range(records)])
    t = np.multiply(t, period)

    # Take appliance power and scale it
    pw = p_true * scale
    pw_pred = p_hat * scale

    # Take power in given interval and get the maximum value
    pw_max = max(pw.max(), pw_pred.max())

    # Plot the figure
    plt.figure(dpi=dpi)

    plt.fill_between(t, p_agg, color='grey', alpha=.4, label='Aggregate power')
    plt.fill_between(t, pw, color=color, alpha=.3, label='Appliance power')
    plt.plot(t, pw_pred, color=color, label='Predicted power')

    plt.legend()

    plt.ylim([0, pw_max * 1.1])
    plt.ylabel('Power (watts)')
    plt.xlabel('Time (minutes)')

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
