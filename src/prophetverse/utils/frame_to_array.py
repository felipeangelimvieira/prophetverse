"""Utilities for converting a pandas DataFrame to JAX/Numpy arrays."""

from typing import Tuple

import numpy as np
import pandas as pd
from jax import numpy as jnp

from .multiindex import iterate_all_series

NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000

__all__ = [
    "convert_index_to_days_since_epoch",
    "series_to_tensor",
    "extract_timetensor_from_dataframe",
    "convert_dataframe_to_tensors",
]


def convert_index_to_days_since_epoch(idx: pd.Index) -> np.array:
    """
    Convert a pandas Index to days since epoch.

    Parameters
    ----------
    idx : pd.Index
        The pandas Index.

    Returns
    -------
    np.ndarray
        The converted array of days since epoch.
    """
    t = idx

    if not (isinstance(t, pd.PeriodIndex) or isinstance(t, pd.DatetimeIndex)):
        return t.values

    if isinstance(t, pd.PeriodIndex):
        t = t.to_timestamp()

    return t.to_numpy(dtype=np.int64) // NANOSECONDS_TO_SECONDS / (3600 * 24.0)


def series_to_tensor(y: pd.DataFrame) -> jnp.ndarray:
    """
    Convert all series of a hierarchical time series to a JAX tensor.

    Parameters
    ----------
    y : pd.DataFrame
        The hierarchical time series.

    Returns
    -------
    jnp.ndarray
        The JAX tensor representing all series.
    """
    names = []
    array = []
    series_len = None

    if y.index.nlevels == 1:
        return jnp.array(y.values).reshape((1, -1, len(y.columns)))

    for idx, series in iterate_all_series(y):
        if series_len is None:
            series_len = len(series)
        if len(series) != series_len:
            raise ValueError(
                f"Series {idx} has length {len(series)}, but expected {series_len}"
            )

        names.append(idx)
        array.append(series.values.reshape((-1, len(y.columns))))
    return jnp.array(array)


def series_to_tensor_or_array(y: pd.DataFrame) -> jnp.ndarray:
    """
    Convert hierarchical (univariate) to three (two) dimensional JAX tensor.

    Parameters
    ----------
    y : pd.DataFrame
        The hierarchical time series.

    Returns
    -------
    jnp.ndarray
        The JAX tensor or the array representing all series.
    """
    if y.index.nlevels == 1:
        return jnp.array(y.values)
    return series_to_tensor(y)


def extract_timetensor_from_dataframe(df: pd.DataFrame) -> jnp.array:
    """
    Extract the time array from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.

    Returns
    -------
    jnp.ndarray
        The JAX tensor representing the time array.
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

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.

    Returns
    -------
    tuple
        A tuple containing the time arrays and the DataFrame arrays as JAX tensors.
    """
    t_arrays = extract_timetensor_from_dataframe(df)

    df_as_arrays = series_to_tensor(df)

    return t_arrays, df_as_arrays
