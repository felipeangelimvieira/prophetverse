from typing import List, Iterable
from sktime.transformations.base import BaseTransformer
import numpy as np
import pandas as pd
from hierarchical_prophet.hierarchical_prophet._time_scaler import TimeScaler
from hierarchical_prophet._utils import convert_index_to_days_since_epoch
class ChangepointMatrix:
    def __init__(self, changepoint_freq, changepoint_range):
        """
        Initialize the ChangepointMatrix.

        Parameters:
            changepoint_freq (list): List of frequencies for each series.
        """
        self.changepoint_freq = changepoint_freq
        self.changepoint_range = changepoint_range

    def fit(self, t):
        """
        Fit the ChangepointMatrix.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ChangepointMatrix: The fitted ChangepointMatrix object.
        """
        self.changepoint_ts = get_changepoint_t(
            t, self.changepoint_freq, self.changepoint_range
        )
        return self

    def transform(self, t):
        """
        Transform the time indices into the changepoint design matrix.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Changepoint design matrix.
        """
        A_tensor = compute_changepoint_design_matrix(t, self.changepoint_ts)
        return A_tensor

    def fit_transform(self, t):
        """
        Fit the ChangepointMatrix and transform the time indices into the changepoint design matrix.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Changepoint design matrix.
        """
        self.fit(t)
        return self.transform(t)

    def changepoint_ts_array(self):
        """
        Get the concatenated changepoint time indices array.

        Returns:
            ndarray: Concatenated changepoint time indices array.
        """
        return np.concatenate(self.changepoint_ts, axis=0)

    @property
    def n_changepoint_per_series(self):
        """
        Get the number of changepoints per series.

        Returns:
            list: List of the number of changepoints per series.
        """
        return [len(x) for x in self.changepoint_ts]


def get_changepoint_t(
    t: np.ndarray, changepoint_freq: Iterable[int], changepoint_range: Iterable[int]
):
    """
    Get the changepoint time indices for each series.

    Parameters:
        t (ndarray): Time indices for each series.
        changepoint_freq (list): List of frequencies for each series.

    Returns:
        list: List of changepoint time indices for each series.
    """

    if t.ndim == 3:
        t = t[0]
    if t.ndim == 2:
        t = t.flatten()
    else:
        raise ValueError("t must be a 2D or 3D array")

    changepoint_ts = []
    for _, (freq, ch_range) in enumerate(zip(changepoint_freq, changepoint_range)):

        changepoint_ts.append(t[:ch_range:freq].flatten())
    return changepoint_ts


def compute_changepoint_design_matrix(t, changepoint_ts):
    """
    Compute the changepoint design matrix.

    Parameters:
        t (ndarray): Time indices for each series.
        changepoint_ts (list): List of changepoint time indices for each series.

    Returns:
        ndarray: Changepoint design matrix.
    """
    n_changepoint_per_series = [len(x) for x in changepoint_ts]
    changepoint_ts = np.concatenate(changepoint_ts)
    changepoint_design_tensor = []
    changepoint_mask_tensor = []
    for i, n_changepoints in enumerate(n_changepoint_per_series):
        expanded_ts = np.tile(t[i].reshape((-1, 1)), (1, sum(n_changepoint_per_series)))
        A = (expanded_ts >= changepoint_ts.reshape((1, -1))).astype(int) * expanded_ts
        cutoff_ts = ((expanded_ts < changepoint_ts.reshape((1, -1))).astype(
            int
        ) * expanded_ts).max(axis=0)
        A = np.clip(A - cutoff_ts, 0, None)

        start_idx = sum(n_changepoint_per_series[:i])
        end_idx = start_idx + n_changepoints
        mask = np.zeros_like(A)
        mask[:, start_idx:end_idx] = 1

        changepoint_design_tensor.append(A)
        changepoint_mask_tensor.append(mask)

    changepoint_design_tensor = np.stack(changepoint_design_tensor, axis=0)
    changepoint_mask_tensor = np.stack(changepoint_mask_tensor, axis=0)
    return changepoint_design_tensor, changepoint_mask_tensor

