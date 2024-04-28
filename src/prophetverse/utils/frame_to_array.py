import re
from operator import le
from typing import Tuple

import numpy as np
import pandas as pd
from jax import numpy as jnp
from sktime.transformations.hierarchical.reconcile import _get_s_matrix

NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000


def get_bottom_series_idx(y):
    """
    Get the index of the bottom series in a hierarchical time series.

    Parameters:
    y (pd.DataFrame): The hierarchical time series.

    Returns:
    pd.Index: The index of the bottom series.
    """
    return _get_s_matrix(y).columns


def get_multiindex_loc(df, idx_values):
    """
    Returns the .loc for the first M levels of a multi-indexed DataFrame for given index values.

    Parameters:
    df (pd.DataFrame): The DataFrame with a multi-index.
    idx_values (tuple or list): Index values for the first M levels.

    Returns:
    pd.DataFrame: The selected data from the DataFrame.
    """
    nlevels = df.index.nlevels

    if isinstance(idx_values[0], tuple):
        nlevels_to_loc = len(idx_values[0])
    else:
        nlevels_to_loc = 1

    levels_to_ignore = list(range(-1, -1 - (nlevels - nlevels_to_loc), -1))

    mask = df.index.droplevel(levels_to_ignore).isin(idx_values)
    return df.loc[mask]


def loc_bottom_series(y):
    """
    Get the bottom series in a hierarchical time series.

    Parameters:
    y (pd.DataFrame): The hierarchical time series.

    Returns:
    pd.DataFrame: The bottom series.
    """
    return get_multiindex_loc(y, get_bottom_series_idx(y))


def iterate_all_series(y):
    """
    Iterate over all series in a hierarchical time series.

    Parameters:
    y (pd.DataFrame): The hierarchical time series.

    Yields:
    tuple: A tuple containing the index and the corresponding series.
    """
    series = _get_s_matrix(y).index
    for idx in series:
        yield idx, y.loc[idx]


def convert_index_to_days_since_epoch(idx: pd.Index) -> np.array:
    """
    Convert a pandas Index to days since epoch.

    Parameters:
    idx (pd.Index): The pandas Index.

    Returns:
    np.ndarray: The converted array of days since epoch.
    """
    t = idx
    if isinstance(t, pd.PeriodIndex):
        t = t.to_timestamp()

    return t.to_numpy(dtype=np.int64) // NANOSECONDS_TO_SECONDS / (3600 * 24.0)


def series_to_tensor(y):
    """
    Convert all series of a hierarchical time series to a JAX tensor.

    Parameters:
    y (pd.DataFrame): The hierarchical time series.

    Returns:
    jnp.ndarray: The JAX tensor representing all series.
    """
    names = []
    array = []
    series_len = None
    for idx, series in iterate_all_series(y):
        if series_len is None:
            series_len = len(series)
        if len(series) != series_len:
            raise ValueError("Series {} has length {}, but expected {}".format(idx, len(series), series_len))
        
        names.append(idx)
        array.append(series.values.reshape((-1, len(y.columns))))
    return jnp.array(array)


def extract_timetensor_from_dataframe(df: pd.DataFrame) -> jnp.array:
    """
    Extract the time array from a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    """

    return series_to_tensor(
        pd.DataFrame(
            index=df.index,
            data={
                "t": convert_index_to_days_since_epoch(df.index.get_level_values(-1))
            },
        )
    )

def convert_dataframe_to_tensors(df: pd.DataFrame) -> Tuple[jnp.array, jnp.array]:
    """
    Convert a pandas DataFrame to JAX tensors.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    tuple: A tuple containing the time arrays and the DataFrame arrays as JAX tensors.
    """
    t_arrays = extract_timetensor_from_dataframe(df)

    df_as_arrays = series_to_tensor(df)

    return t_arrays, df_as_arrays
