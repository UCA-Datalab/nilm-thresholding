import matplotlib.pyplot as plt
import numpy as np


def simple_plot(y_test, y_pred, idx=0):

    # Take just one appliance
    plt_test = y_test[:, :, idx].flatten().copy()
    plt_pred = y_pred[:, :, idx].flatten().copy()

    plt_x = np.arange(plt_test.shape[0])

    plt.figure(dpi=180)
    plt.plot(plt_x, plt_test)
    plt.plot(plt_x, plt_pred, alpha=.75)
    plt.legend(["Test", "Prediction"])
