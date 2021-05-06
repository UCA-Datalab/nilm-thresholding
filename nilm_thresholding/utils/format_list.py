def to_list(x):
    """
    This utility just check if input X is a list
    If not, turns X into a list of one element
    """
    if type(x) is not list:
        return [x]
    else:
        return x


def to_tuple(x):
    """
    This utility just check if input X is a tuple
    If not, turns X into a tuple of one element
    """
    if type(x) is not tuple:
        return (x,)
    else:
        return x


def flatten_list(ls, no_duplicates=True, sort=True):
    """
    Flats a list of sublists.
    """
    flatlist = []
    for sublist in ls:
        for item in sublist:
            flatlist += [item]

    if no_duplicates:
        flatlist = list(set(flatlist))

    if sort:
        flatlist = sorted(flatlist)

    return flatlist
