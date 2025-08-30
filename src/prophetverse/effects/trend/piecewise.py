"""
Piecewise trend models.

This module contains the implementation of piecewise trend models (logistic and linear).
"""

import itertools
from typing import Any, Dict, Tuple, Union, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from sktime.transformations.series.detrend import Detrender

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor

from .base import TrendEffectMixin

__all__ = ["PiecewiseLinearTrend", "PiecewiseLogisticTrend"]


class PiecewiseLinearTrend(TrendEffectMixin, BaseEffect):
    """Piecewise Linear Trend model.

    This model assumes that the trend is piecewise linear, with changepoints
    at regular intervals. The number of changepoints is determined by the
    `changepoint_interval` and `changepoint_range` parameters. The
    `changepoint_interval` parameter specifies the interval between changepoints,
    while the `changepoint_range` parameter specifies the range of the changepoints.

    This implementation is based on the `Prophet`_ library. The initial values (global
    rate and global offset) are suggested using the maximum and minimum values of the
    time series data.


    Parameters
    ----------
    changepoint_interval : int
        The interval between changepoints.
    changepoint_range : int
        The range of the changepoints.
    changepoint_prior_scale : dist.Distribution
        The prior scale for the changepoints.
    offset_prior_scale : float, optional
        The prior scale for the offset. Default is 0.1.
    remove_seasonality_before_suggesting_initial_vals : bool, optional
        If True, remove seasonality before suggesting initial values, using sktime's
        detrender. Default is True.
    global_rate_prior_loc : float, optional
        The prior location for the global rate. Default is suggested
        empirically from data.
    offset_prior_loc : float, optional
        The prior location for the offset. Default is suggested
        empirically from data.


    """

    _tags = {"capability:panel": False}

    def __init__(
        self,
        changepoint_interval: int = 25,
        changepoint_range: int = 0.8,
        changepoint_prior_scale: dist.Distribution = 0.001,
        offset_prior_scale=0.1,
        remove_seasonality_before_suggesting_initial_vals: bool = True,
        global_rate_prior_loc: Optional[float] = None,
        offset_prior_loc: Optional[float] = None,
    ):
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.remove_seasonality_before_suggesting_initial_vals = (
            remove_seasonality_before_suggesting_initial_vals
        )
        self.global_rate_prior_loc = global_rate_prior_loc
        self.offset_prior_loc = offset_prior_loc
        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.

        Set the prior location for the trend.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale : float, optional
            The scale of the timeseries. For multivariate timeseries, this is
            a dataframe. For univariate, it is a simple float.
        """
        super()._fit(X=X, y=y, scale=scale)

        t_scaled = self._index_to_scaled_timearray(
            y.index.get_level_values(-1).unique()
        )
        self._setup_changepoints(t_scaled)
        self._setup_changepoint_prior_vectors(y)
        self._index_names = y.index.names

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> dict:
        """
        Prepare the input data for the piecewise trend model.

        Parameters
        ----------
        X: pd.DataFrame
            The exogenous variables DataFrame.
        fh: pd.Index
            The forecasting horizon as a pandas Index.

        Returns
        -------
        jnp.ndarray
            An array containing the prepared input data.
        """
        return self.get_changepoint_matrix(fh)

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Compute the trend based on the given changepoint matrix.

        Parameters
        ----------
        data: jnp.ndarray
            The changepoint matrix.
        predicted_effects: Dict[str, jnp.ndarray]
            Dictionary of previously computed effects. For the trend, it is an empty
            dict.

        Returns
        -------
        jnp.ndarray
            The computed trend.
        """
        # alias for clarity
        changepoint_matrix = data

        offset = numpyro.sample(
            "offset",
            dist.Normal(self._offset_prior_loc, self._offset_prior_scale),
        )

        changepoint_coefficients = numpyro.sample(
            "changepoint_coefficients",
            dist.Laplace(self._changepoint_prior_loc, self._changepoint_prior_scale),
        )

        trend = (changepoint_matrix) @ changepoint_coefficients + offset

        return trend.reshape((-1, 1))

    def get_changepoint_matrix(self, idx: pd.PeriodIndex) -> jnp.ndarray:
        """
        Return the changepoint matrix for the given index.

        Parameters
        ----------
        idx: pd.PeriodIndex
            The index for which to compute the changepoint matrix.

        Returns
        -------
            jnp.ndarray: The changepoint matrix.
        """
        t_scaled = self._index_to_scaled_timearray(idx)
        changepoint_matrix = _get_changepoint_matrix(t_scaled, self._changepoint_ts)

        return changepoint_matrix

    @property
    def n_changepoint_per_series(self):
        """Get the number of changepoints per series.

        Returns
        -------
        int
            Number of changepoints per series.
        """
        return [len(cp) for cp in self._changepoint_ts]

    @property
    def n_changepoints(self):
        """Get the total number of changepoints.

        Returns
        -------
        int
            Total number of changepoints.
        """
        return len(self._changepoint_ts)

    def _setup_changepoints(self, t_scaled) -> None:
        """
        Set changepoint variables.

        This function has the collateral effect of setting the following attributes:
        - self._changepoint_ts.

        Parameters
        ----------
        t_scaled: jnp.ndarray
            Transformed time index.

        Returns
        -------
            None
        """

        self._changepoint_ts = _get_changepoint_timeindexes(
            t_scaled,
            changepoint_interval=self.changepoint_interval,
            changepoint_range=self.changepoint_range,
        )

        if len(self._changepoint_ts) == 0:
            raise ValueError(
                "No changepoints were generated. Try increasing the changing"
                + f" the changepoint_range. There are {len(t_scaled)} timepoints "
                + f" in the series, changepoint_range is {self.changepoint_range} and "
                + f"changepoint_interval is {self.changepoint_interval}."
            )

    def _setup_changepoint_prior_vectors(self, y: pd.DataFrame) -> None:
        """
        Set up the changepoint prior vectors for the model.

        Parameters
        ----------
        y: pd.DataFrame
            The input DataFrame containing the time series data.

        Returns
        -------
            None
        """
        if self.remove_seasonality_before_suggesting_initial_vals:
            detrender = Detrender()
            y = y - detrender.fit_transform(y)

        self._global_rate, self._offset_prior_loc = (
            self._suggest_global_trend_and_offset(y)
        )
        self._changepoint_prior_loc, self._changepoint_prior_scale = (
            self._get_changepoint_prior_vectors(global_rate=self._global_rate)
        )
        self._offset_prior_scale = self.offset_prior_scale

    def _get_changepoint_prior_vectors(
        self,
        global_rate: jnp.array,
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Return the prior vectors for the changepoint coefficients.

        Parameters
        ----------
        global_rates: jnp.array
            The global rates for each series.

        Returns
        -------
        Tuple[jnp.array, jnp.array]
            A tuple containing the changepoint prior location vector and the
            changepoint prior scale vector.
        """

        def zeros_with_first_value(size, first_value):
            x = jnp.zeros(size)
            x = x.at[0].set(first_value)
            return x

        changepoint_prior_scale_vector = (
            np.ones(self.n_changepoints) * self.changepoint_prior_scale
        )

        changepoint_prior_loc_vector = zeros_with_first_value(
            self.n_changepoints, global_rate
        )

        return jnp.array(changepoint_prior_loc_vector), jnp.array(
            changepoint_prior_scale_vector
        )

    def _suggest_global_trend_and_offset(
        self, y: pd.DataFrame
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Suggest the global trend and offset for the given time series data.

        Parameters
        ----------
        y: pd.DataFrame
            The time series data.

        Returns
        -------
        Tuple[jnp.array, jnp.array]
            A tuple containing the global trend and offset.
        """
        t = self._index_to_scaled_timearray(y.index.get_level_values(-1).unique())

        y_array = series_to_tensor(y)

        global_rate = _enforce_array_if_zero_dim(
            (y_array[:, -1].squeeze() - y_array[:, 0].squeeze())
            / (t[0].squeeze() - t[-1].squeeze())
        )
        offset_loc = y_array[:, 0].squeeze() - global_rate * t[0].squeeze()

        global_rate = global_rate.item()
        offset_loc = offset_loc.item()

        if self.offset_prior_loc is not None:
            offset_loc = self.offset_prior_loc

        if self.global_rate_prior_loc is not None:
            global_rate = self.global_rate_prior_loc

        return global_rate, offset_loc


class PiecewiseLogisticTrend(PiecewiseLinearTrend):
    """
    Piecewise logistic trend model.

    This logistic trend differs from the original Prophet logistic trend in that it
    considers a capacity prior distribution. The capacity prior distribution is used
    to estimate the maximum value that the time series trend can reach.

    It uses internally the piecewise linear trend model, and then applies a logistic
    function to the output of the linear trend model.


    The initial values (global rate and global offset) are suggested using the maximum
    and minimum values of the time series data.


    Parameters
    ----------
    changepoint_interval : int
        The interval between changepoints.
    changepoint_range : int
        The range of the changepoints.
    changepoint_prior_scale : dist.Distribution
        The prior scale for the changepoints.
    offset_prior_scale : float, optional
        The prior scale for the offset. Default is 0.1.
    remove_seasonality_before_suggesting_initial_vals : bool, optional
        If True, remove seasonality before suggesting initial values, using sktime's
        detrender. Default is True.
    capacity_prior : dist.Distribution, optional
        The prior distribution for the capacity. Default is a HalfNormal distribution
        with loc=1.05 and scale=1.
    """

    def __init__(
        self,
        changepoint_interval: int = 25,
        changepoint_range: int = 0.8,
        changepoint_prior_scale: float = 0.001,
        offset_prior_scale=10,
        capacity_prior: dist.Distribution = None,
        remove_seasonality_before_suggesting_initial_vals: bool = True,
        global_rate_prior_loc: Optional[float] = None,
        offset_prior_loc: Optional[float] = None,
    ):

        self.capacity_prior = capacity_prior

        super().__init__(
            changepoint_interval,
            changepoint_range,
            changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            remove_seasonality_before_suggesting_initial_vals=remove_seasonality_before_suggesting_initial_vals,
            global_rate_prior_loc=global_rate_prior_loc,
            offset_prior_loc=offset_prior_loc,
        )

        if capacity_prior is None:
            capacity_prior = dist.TransformedDistribution(
                dist.HalfNormal(
                    0.2,
                ),
                dist.transforms.AffineTransform(loc=1.1, scale=1),
            )
        self._capacity_prior = capacity_prior

    def _suggest_global_trend_and_offset(
        self, y: pd.DataFrame
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Suggest the global trend and offset for the given time series data.

        This implementation considers the max and min values of the timeseries
        to anallytically estimate the global rate and offset.

        Parameters
        ----------
        y: pd.DataFrame
            The input time series data.

        Returns
        -------
        Tuple[jnp.array, jnp.array]
            A tuple containing the suggested global rates
            and offset.
        """
        t_arrays = self._index_to_scaled_timearray(
            y.index.get_level_values(-1).unique()
        )
        y_arrays = series_to_tensor(y)

        if hasattr(self._capacity_prior, "loc"):
            capacity_prior_loc = self._capacity_prior.loc
        else:
            capacity_prior_loc = y_arrays.max() * 1.05

        global_rates, offset = _suggest_logistic_rate_and_offset(
            t=t_arrays.squeeze(),
            y=y_arrays.squeeze(),
            capacities=capacity_prior_loc,
        )

        return global_rates.item(), offset.item()

    def _predict(  # type: ignore[override]
        self, data: Any, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """
        Compute the trend for the given changepoint matrix.

        Parameters
        ----------
        changepoint_matrix: jnp.ndarray
            The changepoint matrix.

        Returns
        -------
        jnp.ndarray
            The computed trend.
        """
        with numpyro.plate("series", self.n_series, dim=-3):
            capacity = numpyro.sample("capacity", self._capacity_prior)

        trend = super()._predict(data=data, predicted_effects=predicted_effects)

        if self.n_series == 1:
            capacity = capacity.squeeze()

        trend = capacity / (1 + jnp.exp(-trend))
        return trend


def _to_list_if_scalar(x, size=1):
    """
    Convert a scalar value to a list of the same value repeated `size` times.

    Parameters
    ----------
    x: scalar or array-like
        The input value to be converted.
    size: int, optional
        The number of times to repeat the value in the list. Default is 1.

    Returns
    -------
    list
        A list containing the input value repeated `size` times if `x` is
        a scalar, otherwise returns `x` unchanged.
    """
    if np.isscalar(x):
        return [x] * size
    return x


def _enforce_array_if_zero_dim(x):
    """
    Reshapes the input array `x` to have at least one dimension if it has zero dim.

    Parameters
    ----------
    x: array-like
        The input array.

    Returns
    -------
    array-like
        The reshaped array.
    """
    if x.ndim == 0:
        return x.reshape(1)
    return x


def _suggest_logistic_rate_and_offset(
    t: np.ndarray, y: np.ndarray, capacities: Union[float, np.ndarray]
):
    """
    Suggest the logistic rate and offset based on the given time series data.

    Parameters
    ----------
    t: ndarray
        The time values of the time series data.
    y: ndarray
        The observed values of the time series data.
    capacities: float or ndarray
        The capacity or capacities of the time series data.

    Returns
    -------
    m: jnp.ndarray
        The suggested offset.
    k: jnp.ndarray
        The suggested logistic rate.
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

    if np.any(y0 == 0):
        y0 = 1e-8

    r0 = capacities / y0
    r1 = capacities / y1

    L0 = np.log(r0 - 1)
    L1 = np.log(r1 - 1)

    if np.any(np.isnan(L0)) or np.any(np.isnan(L1)):
        error_str = (
            "L0 is {}, L1 is {}. At least one of them has NaN,"
            + "the input parameters were: t: {}, y: {}, capacities: {}"
        )
        raise ValueError(error_str.format(L0, L1, t, y, capacities))

    k = (L1 - L0) / T
    m = -(L1 + k * t1)

    return k, m


def _get_changepoint_matrix(t: jnp.ndarray, changepoint_t: jnp.array) -> jnp.ndarray:
    """
    Generate a changepoint matrix based on the time indexes and changepoint indexes.

    Parameters
    ----------
    t: jnp.ndarray
        array with timepoints of shape (n, 1) preferably
    changepoint_t: jnp.array
        array with changepoint timepoints of shape (n_changepoints,)

    Returns
    -------
    jnp.ndarray
        changepoint matrix of shape (n, n_changepoints)
    """
    expanded_ts = jnp.tile(t.reshape((-1, 1)), (1, len(changepoint_t)))
    A = (expanded_ts >= changepoint_t.reshape((1, -1))).astype(int) * expanded_ts
    cutoff_ts = changepoint_t.reshape((1, -1))
    A = jnp.clip(A - cutoff_ts, 0, None)
    return A


def _get_changepoint_timeindexes(
    t: jnp.ndarray, changepoint_interval: int, changepoint_range: float = 0.90
) -> jnp.array:
    """
    Return an array of time indexes for changepoints based on the given parameters.

    Parameters
    ----------
    t: jnp.ndarray
        The array of time values.
    changepoint_interval: int
        The interval between changepoints.
    changepoint_range: float, optional
        The range of changepoints. Defaults to 0.90. If greater than 1, it is
        interpreted as then number of timepoints.If less than zero, it is interpreted as
        number of timepoints from the end of the time series.

    Returns
    -------
    jnp.array
        An array of time indexes for changepoints.
    """
    if changepoint_range < 1 and changepoint_range > 0:
        max_t = t.max() * changepoint_range
    elif changepoint_range >= 1:
        max_t = changepoint_range
    else:
        max_t = t.max() + changepoint_range

    changepoint_t = jnp.arange(0, max_t, changepoint_interval)
    return changepoint_t
