import re
from unicodedata import normalize

import numpy as np

APPLIANCE_NAMES = {
    "freezer": "fridge",
    "fridgefreezer": "fridge",
    "washerdryer": "washingmachine"
}


def deaccent(string, remove_dieresis_u=True):
    """
    Eliminate all the accents from string, keeping the ñ.
    Optionally removes dieresis in ü.
    Parameters
    ----------
    string : str
    remove_dieresis_u : bool, default=True
        If True, it removes the dieresis on the Ü and ü
    Returns
    -------
    string_deaccent : str
        Deaccent version of string.
    Extra Info
    ----------
    https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
    """

    # -> NFD
    string_decomposed = normalize("NFD", string)
    # delete accents
    if remove_dieresis_u:
        # keep the tilde on the n (ñ -> n)
        string_deaccent = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
            r"\1", string_decomposed, 0, re.I)
    else:
        # keep the tilde on the n (ñ -> n) and dieresis on the u (ü -> u)
        string_deaccent = re.sub(
            r"([^nu\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f]))|u(?!\u0308(?![\u0300-\u036f])))[\u0300-\u036f]+",
            r"\1", string_decomposed, 0, re.I)
    # -> NFC
    string_deaccent = normalize("NFC", string_deaccent)
    assert len(string_deaccent) == len(
        string), "String has different length after applying deaccent."
    return string_deaccent


def homogenize_string(string, remove_dieresis_u=True):
    """
    Lowercases and eliminates all the accents from string, keeping the ñ.
    Optionally removes dieresis in ü.
    Eliminate spaces.
    Parameters
    ----------
    string : str
    remove_dieresis_u : bool, default=True
        If True, it removes the dieresis on the Ü and ü
    Returns
    -------
    string_deaccent : str
        Lowercase and deaccent version of the input entity.
    """
    if string is np.nan:
        string = "NaN"
    assert isinstance(string, str), f"{string} is not a str object"
    string_low = string.strip().lower()
    string_low = string_low.replace("_", " ")
    string_low = string_low.replace(" ", "")
    string_low_deaccent = deaccent(
        string_low, remove_dieresis_u=remove_dieresis_u)
    return string_low_deaccent
