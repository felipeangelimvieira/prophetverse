"""Utilities for working with multi-indexed DataFrames."""

import pandas as pd

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


def _get_s_matrix(X):
    """Determine the summation "S" matrix.

    Reconciliation methods require the S matrix, which is defined by the
    structure of the hierarchy only. The S matrix is inferred from the input
    multi-index of the forecasts and is used to sum bottom-level forecasts
    appropriately.

    Please refer to [1]_ for further information.

    Parameters
    ----------
    X :  Panel of mtype pd_multiindex_hier

    Returns
    -------
    s_matrix : pd.DataFrame with rows equal to the number of unique nodes in
        the hierarchy, and columns equal to the number of bottom level nodes only,
        i.e. with no aggregate nodes. The matrix indexes is inherited from the
        input data, with the time level removed.

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html
    """
    # get bottom level indexes
    bl_inds = (
        X.loc[~(X.index.get_level_values(level=-2).isin(["__total"]))]
        .index.droplevel(level=-1)
        .unique()
    )
    # get all level indexes
    al_inds = X.droplevel(level=-1).index.unique()

    # set up matrix
    s_matrix = pd.DataFrame(
        [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
        index=al_inds,
    )
    s_matrix.columns = bl_inds

    # now insert indicator for bottom level
    for i in s_matrix.columns:
        s_matrix.loc[s_matrix.index == i, i] = 1.0

    # now for each unique column add aggregate indicator
    for i in s_matrix.columns:
        if s_matrix.index.nlevels > 1:
            # replace index with totals -> ("nodeA", "__total")
            agg_ind = list(i)[::-1]
            for j in range(len(agg_ind)):
                agg_ind[j] = "__total"
                # insert indicator
                s_matrix.loc[tuple(agg_ind[::-1]), i] = 1.0
        else:
            s_matrix.loc["__total", i] = 1.0

    # drop new levels not present in original matrix
    s_matrix = s_matrix.loc[s_matrix.index.isin(al_inds)]

    return s_matrix
