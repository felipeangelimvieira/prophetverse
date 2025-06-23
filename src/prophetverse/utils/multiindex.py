"""Utilities for working with multi-indexed DataFrames."""

import pandas as pd
from sktime import __version__

if __version__ < "0.37.1":
    from sktime.transformations.hierarchical.reconcile import _get_s_matrix
else:
    from sktime.transformations.hierarchical.reconcile._reconcile import _get_s_matrix


__all__ = [
    "get_bottom_series_idx",
    "get_multiindex_loc",
    "loc_bottom_series",
    "iterate_all_series",
    "reindex_time_series",
]


def get_bottom_series_idx(y):
    """
    Get the index of the bottom series in a hierarchical time series.

    Parameters
    ----------
    y : pd.DataFrame
        The hierarchical time series.

    Returns
    -------
    pd.Index
        The index of the bottom series.
    """
    if y.index.nlevels == 1:
        raise ValueError("y must be a multi-index DataFrame")
    if y.index.nlevels == 2:
        return pd.Index([x for x in y.index.droplevel(-1).unique() if x != "__total"])
    series = pd.Index([x for x in y.index.droplevel(-1).unique() if x[-1] != "__total"])
    return series


def loc_bottom_series(y):
    """
    Get the bottom series in a hierarchical time series.

    Parameters
    ----------
    y : pd.DataFrame
        The hierarchical time series.

    Returns
    -------
    pd.DataFrame
        The bottom series.
    """
    if y.index.nlevels == 1:
        return y
    return get_multiindex_loc(y, get_bottom_series_idx(y))


def get_multiindex_loc(df, idx_values):
    """
    Return the .loc for the first M levels of a multi-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with a multi-index.
    idx_values : tuple or list
        Index values for the first M levels.

    Returns
    -------
    pd.DataFrame
        The selected data from the DataFrame.
    """
    nlevels = df.index.nlevels

    if isinstance(idx_values[0], tuple):
        nlevels_to_loc = len(idx_values[0])
    else:
        nlevels_to_loc = 1

    levels_to_ignore = list(range(-1, -1 - (nlevels - nlevels_to_loc), -1))

    mask = df.index.droplevel(levels_to_ignore).isin(idx_values)
    return df.loc[mask]


def reindex_time_series(df, new_time_index):
    """
    Reindex the time index level (-1) of a multi-index DataFrame with a new time index.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with a multi-level index.
    new_time_index : pd.Index or similar
        The new time index to apply to the DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the updated time index.
    """
    if df.index.nlevels == 1:
        return df.reindex(new_time_index)

    # Extract the current indices except for the time index
    levels = df.index.levels[:-1]
    names = df.index.names[:-1]

    # Create a new multi-index combining the existing levels with the new time index
    new_index = pd.MultiIndex.from_product(
        levels + [new_time_index], names=names + [df.index.names[-1]]
    )

    # Reindex the DataFrame using the new index
    return df.reindex(new_index)


def iterate_all_series(y):
    """
    Iterate over all series in a hierarchical time series.

    Parameters
    ----------
    y : pd.DataFrame
        The hierarchical time series.

    Yields
    ------
    tuple
        A tuple containing the index and the corresponding series.
    """
    series = _get_s_matrix(y).index
    for idx in series:
        yield idx, y.loc[idx]
