from collections import defaultdict


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
        return x,
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


def merge_dict_list(dict_list):
    d = defaultdict(dict)
    for l in dict_list:
        for elem in l:
            d[elem].update(l[elem])

    return d