def print_basic_statistics(x, name):
    """
    Prints X size, mean, maximum and minimum values.

    Parameters
    ----------
    x : numpy array
    name : str
        Name of X

    """
    print("------------------------------------------------------------------")
    print(f"{name} records:", x.size)
    print(f"{name} sequences:", x.shape[0])
    print(f"{name} mean:", x.mean())
    print(f"{name} max:", x.max())
    print(f"{name} min:", x.min())


def print_appliance_statistics(y, name, appliances):
    """

    Parameters
    ----------
    y : numpy.array
    name : str
        Name of Y
    appliances : list
        List of appliances, sorted according to Y

    """
    print("------------------------------------------------------------------")
    y_mean = y.mean(axis=(0, 1))
    y_max = y.max(axis=(0, 1))

    for idx, app in enumerate(appliances):
        print(f"{name} {app} mean:", y_mean[idx])
        print(f"{name} {app} max:", y_max[idx])

