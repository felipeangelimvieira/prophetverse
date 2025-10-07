"""Regex utilities to facilitate the definition of columns for effects."""

__all__ = ["starts_with", "exact", "no_input_columns", "ends_with", "contains"]

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


def ends_with(suffixes):
    """
    Return a regular expression pattern that matches strings ending with given suffixes.

    Parameters
    ----------
    suffixes: str or list of str
        A string or a list of strings representing the suffixes to match.

    Returns
    -------
    str
        A regular expression pattern that matches strings ending with any of the
        given suffixes.
    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    return rf"(?:{'|'.join(suffixes)})$"


def contains(patterns):
    """
    Return a regular expression pattern that matches strings containing given patterns.

    Parameters
    ----------
    patterns: str or list of str
        A string or a list of strings, where each string is a pattern to be searched for.

    Returns
    -------
    str
        A regular expression pattern that matches strings containing any of the
        given patterns.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    return rf"(?:{'|'.join(patterns)})"


def filter_columns(df, pattern):
    """
    Return a list of column names in the DataFrame that match the given regex pattern.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame whose columns are to be matched against the regex pattern.
    pattern: str
        The regular expression pattern to match column names.

    Returns
    -------
    list
        A list of column names that match the given regex pattern.
    """
    columns = df.columns

    if pattern is None:
        return None

    if isinstance(pattern, list):
        return columns[columns.isin(pattern)].tolist()

    columns = columns.astype(str)
    return columns[columns.str.match(pattern)].tolist()
