"""Regex utilities to facilitate the definition of columns for effects."""

__all__ = ["starts_with", "exact", "no_input_columns"]

no_input_columns = r"^$"


def starts_with(prefixes):
    """
    Return a regular expression pattern that matches strings starting given prefixes.

    Parameters
    ----------
    prefixes: list
        A list of strings representing the prefixes to match.

    Returns
    -------
    str
        A regular expression pattern that matches strings starting with any of the
        given prefixes.
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return rf"^(?:{'|'.join(prefixes)})"


def exact(string):
    """
    Return a regular expression pattern that matches the exact given string.

    Parameters
    ----------
    string: str
        The string to match exactly.

    Returns
    -------
    str
        A regular expression pattern that matches the exact given string.
    """
    return rf"^{string}$"
