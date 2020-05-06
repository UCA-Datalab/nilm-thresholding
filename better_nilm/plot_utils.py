import matplotlib.pyplot as plt
import numpy as np


def comparison_plot(y_test, y_pred, idx=0, savefig=None):
    """
    Plots the evolution of real and predicted appliance load values,
    assuming all are consecutive.

    Parameters
    ----------
    y_test : numpy.array
    y_pred : numpy.array
    idx : int
        Appliance index
    savefig : str
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
