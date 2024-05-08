import numpyro
import numpyro.distributions as dist
from prophetverse.utils.frame_to_array import (
    convert_index_to_days_since_epoch,
    series_to_tensor,
)
import numpy as np

import jax.numpy as jnp
import pandas as pd
from .changepoint import get_changepoint_matrix, get_changepoint_timeindexes
from abc import ABC, abstractmethod
import numpy as np
from typing import List


class TrendModel(ABC):

    def initialize(self, y: pd.DataFrame):
        # Set time scale
        t_days = convert_index_to_days_since_epoch(y.index.get_level_values(-1).unique())
        self.t_scale = (t_days[1:] - t_days[:-1]).mean()
        self.t_start = t_days.min() / self.t_scale
        if y.index.nlevels > 1:
            self.n_series = y.index.droplevel(-1).nunique()
        else:
            self.n_series = 1
        

    @abstractmethod
    def prepare_input_data(self, idx: pd.PeriodIndex) -> dict:
        """
        Returns a dictionary containing the data needed for the trend model.
        For example, given t, a possible implementation would return
        {
            "changepoint_matrix": jnp.array([[1, 0], [0, 1]]),
        }

        """
        ...

    @abstractmethod
    def compute_trend(self, **kwargs):
        """
        Computes
        """
        ...

    def _index_to_scaled_timearray(self, idx):
        t_days = convert_index_to_days_since_epoch(idx)
        return (t_days) / self.t_scale - self.t_start

    def __call__(self, **kwargs):
        return self.compute_trend(**kwargs)



class PiecewiseLinearTrend(TrendModel):

    def __init__(
        self,
        changepoint_interval: int,
        changepoint_range: int,
        changepoint_prior_scale: dist.Distribution,
        offset_prior_scale=0.1,
        squeeze_if_single_series: bool = True,
        **kwargs
    ):
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.squeeze_if_single_series = squeeze_if_single_series
        super().__init__(**kwargs)


    def initialize(self, y: pd.DataFrame):

        super().initialize(y)
        t_scaled = self._index_to_scaled_timearray(
            y.index.get_level_values(-1).unique()
        )
        self._setup_changepoints(t_scaled)
        self._setup_changepoint_prior_vectors(y)

    @property
    def n_changepoint_per_series(self):
        """Get the number of changepoints per series.

        Returns:
            int: Number of changepoints per series.
        """
        return [len(cp) for cp in self._changepoint_ts]

    @property
    def n_changepoints(self):
        """Get the total number of changepoints.

        Returns:
            int: Total number of changepoints.
        """
        return sum(self.n_changepoint_per_series)

    def _get_multivariate_changepoint_matrix(self, t_scaled):
        """
        Get the changepoint matrix. The changepoint matrix has shape (n_series, n_timepoints, total number of changepoints for all series).
        A mask is applied so that for index i at dim 0, only the changepoints for series i are non-zero at dim -1.

        Args:
            t_scaled (ndarray): Transformed time index.

        Returns:
            ndarray: The changepoint matrix.
        """
        changepoint_ts = np.concatenate(self._changepoint_ts)
        changepoint_design_tensor = []
        changepoint_mask_tensor = []
        for i, n_changepoints in enumerate(self.n_changepoint_per_series):
            A = get_changepoint_matrix(t_scaled, changepoint_ts)

            start_idx = sum(self.n_changepoint_per_series[:i])
            end_idx = start_idx + n_changepoints
            mask = np.zeros_like(A)
            mask[:, start_idx:end_idx] = 1

            changepoint_design_tensor.append(A)
            changepoint_mask_tensor.append(mask)

        changepoint_design_tensor = np.stack(changepoint_design_tensor, axis=0)
        changepoint_mask_tensor = np.stack(changepoint_mask_tensor, axis=0)
        return changepoint_design_tensor * changepoint_mask_tensor

    def _setup_changepoints(self, t_scaled):
        """
        Setup changepoint variables and transformer.

        This function has the collateral effect of setting the following attributes:
        - self._changepoint_ts

        Args:
            t_arrays (ndarray): Transformed time index.

        Returns:
            None
        """
        changepoint_intervals = to_list_if_scalar(
            self.changepoint_interval, self.n_series
        )
        changepoint_ranges = to_list_if_scalar(
            self.changepoint_range or -self.changepoint_interval, self.n_series
        )

        changepoint_ts = []
        for changepoint_interval, changepoint_range in zip(
            changepoint_intervals, changepoint_ranges
        ):
            changepoint_ts.append(
                get_changepoint_timeindexes(
                    t_scaled,
                    changepoint_interval=changepoint_interval,
                    changepoint_range=changepoint_range,
                )
            )

        self._changepoint_ts = changepoint_ts

    def _setup_changepoint_prior_vectors(self, y: pd.DataFrame):
        self.global_rates, self.offset_loc = self._suggest_global_trend_and_offset(y)
        self._changepoint_prior_loc, self._changepoint_prior_scale = (
            self._get_changepoint_prior_vectors(global_rates=self.global_rates)
        )

    def get_changepoint_matrix(self, idx: pd.PeriodIndex):

        t_scaled = self._index_to_scaled_timearray(idx)
        changepoint_matrix = self._get_multivariate_changepoint_matrix(t_scaled)

        # If only one series, remove the first dimension
        if changepoint_matrix.shape[0] == 1:
            if self.squeeze_if_single_series:
                changepoint_matrix = changepoint_matrix[0]

        return changepoint_matrix

    def _get_changepoint_prior_vectors(
        self,
        global_rates: jnp.array,
    ):
        """

        Returns the prior vectors for the changepoint coefficients.
        Returns:
            None
        """

        n_series = len(self.n_changepoint_per_series)

        def zeros_with_first_value(size, first_value):
            x = jnp.zeros(size)
            x.at[0].set(first_value)
            return x

        changepoint_prior_scale_vector = np.concatenate(
            [
                np.ones(n_changepoint) * cur_changepoint_prior_scale
                for n_changepoint, cur_changepoint_prior_scale in zip(
                    self.n_changepoint_per_series,
                    to_list_if_scalar(self.changepoint_prior_scale, n_series),
                )
            ]
        )

        changepoint_prior_loc_vector = np.concatenate(
            [
                zeros_with_first_value(n_changepoint, estimated_global_rate)
                for n_changepoint, estimated_global_rate in zip(
                    self.n_changepoint_per_series, global_rates
                )
            ]
        )

        return jnp.array(changepoint_prior_loc_vector), jnp.array(
            changepoint_prior_scale_vector
        )

    def prepare_input_data(self, idx: pd.PeriodIndex) -> dict:

        return {"changepoint_matrix": self.get_changepoint_matrix(idx)}

    def _suggest_global_trend_and_offset(self, y: pd.DataFrame):

        t = self._index_to_scaled_timearray(y.index.get_level_values(-1).unique())

        y_array = series_to_tensor(y)

        global_rate = enforce_array_if_zero_dim(
            (y_array[:, -1].squeeze() - y_array[:, 0].squeeze())
            / (t[0].squeeze() - t[-1].squeeze())
        )
        offset_loc = y_array[:, 0].squeeze() - global_rate * t[0].squeeze()

        return global_rate, offset_loc

    def compute_trend(self, changepoint_matrix: jnp.ndarray):

        offset = numpyro.sample(
            "offset", dist.Normal(self.offset_loc, self.offset_prior_scale)
        )

        changepoint_coefficients = numpyro.sample(
            "changepoint_coefficients", dist.Laplace(self._changepoint_prior_loc, self._changepoint_prior_scale)
        )

        # If multivariate
        if changepoint_matrix.ndim == 3:
            changepoint_coefficients = changepoint_coefficients.reshape((1, -1, 1))
            offset = offset.reshape((-1, 1, 1))

        trend = (changepoint_matrix) @ changepoint_coefficients + offset

        if trend.ndim == 1:
            trend = trend.reshape((-1, 1))
        return trend


class PiecewiseLogisticTrend(PiecewiseLinearTrend): 

    def __init__(
        self,
        
        changepoint_interval: int,
        changepoint_range: int,
        changepoint_prior_scale: float,
        offset_prior_scale=10,
        capacity_prior : dist.Distribution = None,
        **kwargs
    ):

        if capacity_prior is None:
            capacity_prior = dist.TransformedDistribution(
                dist.HalfNormal(
                    0.2,
                ),
                dist.transforms.AffineTransform(
                    loc=1.05, scale=1
                ),
            )

        self.capacity_prior = capacity_prior

        super().__init__(
            changepoint_interval,
            changepoint_range,
            changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            squeeze_if_single_series=False,
            **kwargs
        )

    def _suggest_global_trend_and_offset(self, y: pd.DataFrame):

        t_arrays = self._index_to_scaled_timearray(y.index.get_level_values(-1).unique())
        y_arrays = series_to_tensor(y)

        if hasattr(self.capacity_prior, "loc"):
            capacity_prior_loc = self.capacity_prior.loc
        else:
            capacity_prior_loc = 1.05

        global_rates, offset = suggest_logistic_rate_and_offset(
                t=t_arrays.squeeze(),
                y=y_arrays.squeeze(),
                capacities=capacity_prior_loc,
        )

        return global_rates, offset

    def compute_trend(self, changepoint_matrix: jnp.ndarray):

        with numpyro.plate("series", self.n_series, dim=-3):
            capacity = numpyro.sample(
                "capacity", self.capacity_prior
            )
        
        trend = super().compute_trend(changepoint_matrix)
        if self.n_series == 1:
            capacity = capacity.squeeze()
            
        trend = capacity / (1 + jnp.exp(-trend))
        return trend


def to_list_if_scalar(x, size=1):
    """
    Converts a scalar value to a list of the same value repeated `size` times.

    Args:
        x (scalar or array-like): The input value to be converted.
        size (int, optional): The number of times to repeat the value in the list. Default is 1.

    Returns:
        list: A list containing the input value repeated `size` times if `x` is a scalar, otherwise returns `x` unchanged.
    """
    if np.isscalar(x):
        return [x] * size
    return x


def enforce_array_if_zero_dim(x):
    """
    Reshapes the input array `x` to have at least one dimension if it has zero dimensions.

    Args:
        x (array-like): The input array.

    Returns:
        array-like: The reshaped array.

    """
    if x.ndim == 0:
        return x.reshape(1)
    return x


def suggest_logistic_rate_and_offset(
    t: np.ndarray, y: np.ndarray, capacities: float or np.ndarray
):
    """
    Suggests the logistic rate and offset based on the given time series data.

    Parameters:
        t (ndarray): The time values of the time series data.
        y (ndarray): The observed values of the time series data.
        capacities (float or ndarray): The capacity or capacities of the time series data.

    Returns:
        m (ndarray): The suggested offset.
        k (ndarray): The suggested logistic rate.

    """

    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim == 3:
        # Shape here would be (n_series, n_samples, 1)
        y = y.squeeze()
    if t.ndim == 1:
        t = t.reshape(1, -1)
    elif t.ndim == 3:
        # Shape here would be (n_series, n_samples, 1)
        t = t.squeeze()

    i0, i1 = t.argmin(axis=1), t.argmax(axis=1)
    t0, t1 = t[:, i0].flatten(), t[:, i1].flatten()
    T = t0 - t1
    y0, y1 = y[:, i0].flatten(), y[:, i1].flatten()

    r0 = capacities / y0
    r1 = capacities / y1

    L0 = np.log(r0 - 1)
    L1 = np.log(r1 - 1)

    
    k = (L1 - L0) / T
    m = - (L1 + k * t1)

    return k, m
