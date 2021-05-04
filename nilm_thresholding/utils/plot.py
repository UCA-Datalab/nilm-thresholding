import matplotlib.pyplot as plt


def plot_real_data(
    dict_pred: dict,
    figsize: tuple = (10, 6),
    idx_start: int = 0,
    idx_end: int = 480,
    savefig: str = None,
):
    """Plot the aggregated power load and each appliances' consumption"""
    # List appliances
    list_app = list(dict_pred.keys())
    list_app.remove("aggregate")
    fig = plt.figure(figsize=figsize)
    axs = fig.add_axes()
    axs.plot(dict_pred["aggregate"][idx_start:idx_end], label="Aggregate")
    for app in list_app:
        name = app.capitalize().replace("_", " ")
        axs.plot(dict_pred[app]["power"][idx_start:idx_end], label=name)
    axs.legend()
    axs.set_ylabel("Power (watts)")
    if savefig is None:
        return fig, axs
    else:
        plt.savefig(savefig)
        plt.close()
