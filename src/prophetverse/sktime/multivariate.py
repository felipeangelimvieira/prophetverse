import itertools
import logging
import re
import functools
from typing import Dict, List, Tuple, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from typing import Callable
import pandas as pd
from jax import lax, random
from collections import OrderedDict
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import _get_s_matrix


from prophetverse.utils.frame_to_array import (
    convert_index_to_days_since_epoch,
    extract_timetensor_from_dataframe,
    loc_bottom_series,
    series_to_tensor,
)
from prophetverse.utils.multiindex import reindex_time_series
from prophetverse.utils.logistic import suggest_logistic_rate_and_offset
from prophetverse.sktime.base import (
    BaseBayesianForecaster,
    ExogenousEffectMixin,
    init_params,
)


from ._expand_column_per_level import (
    ExpandColumnPerLevel,
)

from prophetverse.changepoint import (
    get_changepoint_matrix,
    get_changepoint_timeindexes,
)

from prophetverse.models.multivariate_model._model import model
from prophetverse.effects import LinearEffect, LinearHeterogenousPriorsEffect

logger = logging.getLogger("sktime-numpyro")


class HierarchicalProphet(ExogenousEffectMixin, BaseBayesianForecaster):
    """A class that represents a Bayesian hierarchical time series forecasting model based on the Prophet algorithm.

    This class forecasts all series in a hierarchy at once, using a MultivariateNormal as the likelihood function, and
    LKJ priors for the correlation matrix.

    This class may be interesting if you want to fit shared coefficients across series. By default, all coefficients are
    obtained exclusively for each series, but this can be changed through the `shared_coefficients` parameter.

    Args:
        changepoint_interval (int): The number of points between each potential changepoint.
        changepoint_range (float): Proportion of the history in which trend changepoints will be estimated.
                                   If a float between 0 and 1, the range will be that proportion of the history.
                                   If an int, the range will be that number of points. A negative int indicates number of points
                                   counting from the end of the history.
        changepoint_prior_scale (float): Parameter controlling the flexibility of the automatic changepoint selection.
        capacity_prior_scale (float): Scale parameter for the capacity prior. Defaults to 0.2.
        capacity_prior_loc (float): Location parameter for the capacity prior. Defaults to 1.1.
        trend (str): Type of trend. Either "linear" or "logistic". Defaults to "linear".
        feature_transformer (BaseTransformer or None): A transformer to preprocess the exogenous features. Defaults to None.
        exogenous_effects (list or None): List of exogenous effects to include in the model. Defaults to None.
        default_effect_mode (str): Default mode for exogenous effects. Either "multiplicative" or "additive". Defaults to "multiplicative".
        default_exogenous_prior (tuple): Default prior distribution for exogenous effects. See numpyro.distributions for options. Defaults to ("Normal", 0, 1).
        shared_features (list): List of shared features across series. Defaults to an empty list.
        mcmc_samples (int): Number of MCMC samples to draw. Defaults to 2000.
        mcmc_warmup (int): Number of warmup steps for MCMC. Defaults to 200.
        mcmc_chains (int): Number of MCMC chains. Defaults to 4.
        inference_method (str): Inference method to use. Either "map" or "mcmc". Defaults to "map".
        optimizer_name (str): Name of the optimizer to use. Defaults to "Adam".
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer. Defaults to {"step_size": 1e-4}.
        optimizer_steps (int): Number of optimization steps. Defaults to 100_000.
        noise_scale (float): Scale parameter for the noise. Defaults to 0.05.
        correlation_matrix_concentration (float): Concentration parameter for the correlation matrix. Defaults to 1.0.
        rng_key (jax.random.PRNGKey): Random number generator key. Defaults to random.PRNGKey(24).
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.Series",
            "pd_multiindex_hier",
            "pd-multiindex",
        ],
        "X_inner_mtype": [
            "pd_multiindex_hier",
            "pd-multiindex",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "fit_is_empty": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=None,
        changepoint_prior_scale=0.1,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        trend="linear",
        feature_transformer: BaseTransformer = None,
        exogenous_effects=None,
        default_effect_mode="multiplicative",
        default_exogenous_prior=("Normal", 0, 1),
        shared_features=[],
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs={"step_size": 1e-4},
        optimizer_steps=100_000,
        noise_scale=0.05,
        correlation_matrix_concentration=1.0,
        rng_key=random.PRNGKey(24),
    ):

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.noise_scale = noise_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend
        self.default_exogenous_prior = default_exogenous_prior
        self.shared_features = shared_features
        self.feature_transformer = feature_transformer
        self.correlation_matrix_concentration = correlation_matrix_concentration

        super().__init__(
            rng_key=rng_key,
            default_effect_mode=default_effect_mode,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            exogenous_effects=exogenous_effects,
            default_exogenous_prior=default_exogenous_prior,
        )

        self.model = model
        self._validate_hyperparams()

    def _validate_hyperparams(self):
        """
        Validate the hyperparameters of the HierarchicalProphet forecaster.
        """
        if self.changepoint_interval <= 0:
            raise ValueError("changepoint_interval must be greater than 0.")

        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be greater than 0.")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")
        if self.capacity_prior_scale <= 0:
            raise ValueError("capacity_prior_scale must be greater than 0.")
        if self.capacity_prior_loc <= 0:
            raise ValueError("capacity_prior_loc must be greater than 0.")
        if self.default_effect_mode not in ["multiplicative", "additive"]:
            raise ValueError(
                'seasonality_mode must be either "multiplicative" or "additive".'
            )
        if self.trend not in ["linear", "logistic"]:
            raise ValueError('trend must be either "linear" or "logistic".')

    def _get_fit_data(self, y, X, fh):
        """
        Prepare the data for the NumPyro model.

        Args:
            y (pd.DataFrame): Training target time series.
            X (pd.DataFrame): Training exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: A dictionary containing the model data.
        """

        # Handling series without __total indexes
        self.aggregator_ = Aggregator()
        self.original_y_indexes_ = y.index
        y = self.aggregator_.fit_transform(y)
        self.full_y_indexes_ = y.index

        # Hierarchy matrix (S Matrix)
        self.hierarchy_matrix = jnp.array(_get_s_matrix(y).values)

        # Updating internal _y of sktime because BaseBayesianForecaster uses it to convert
        # Forecast Horizon into multiindex correcly
        self._y = y

        # Convert inputs to array, including the time index
        y_bottom = loc_bottom_series(y)
        y_bottom_arrays = series_to_tensor(y_bottom)
        t_arrays = extract_timetensor_from_dataframe(y_bottom)

        # Setup time scale
        self._set_time_scale(t_arrays)
        t_scaled = self._time_scaler.scale(t_arrays)
        # We use a single time array for all series, with shape (n_timepoints, 1)
        t_scaled = t_scaled[0].reshape((-1, 1))
        # Changepoints
        self._setup_changepoints(t_scaled=t_scaled)
        changepoint_matrix = self._get_changepoint_matrix(t_scaled)

        # Exog variables

        # If no exogenous variables, create empty DataFrame
        # Else, aggregate exogenous variables and transform them
        if (X is None or X.columns.empty) and self.feature_transformer is not None:
            X = pd.DataFrame(index=y.index)
        if self.feature_transformer is not None:
            X = self.feature_transformer.fit_transform(X)
        self._has_exogenous_variables = X is not None and not X.columns.empty

        if self._has_exogenous_variables:

            self.expand_columns_transformer_ = ExpandColumnPerLevel(
                X.columns.difference(self.shared_features)
            ).fit(X)
            X = X.loc[y.index]
            X = self.expand_columns_transformer_.transform(X)

            self._set_custom_effects(feature_names=X.columns)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))

        else:
            self._exogenous_effects_and_columns = {}
            exogenous_data = {}

        self.fit_and_predict_data_ = {
            "trend_mode": self.trend,
            "exogenous_effects": self.exogenous_effect_dict,
            "init_trend_params": self._get_trend_sample_func(
                t_arrays=t_scaled, y_arrays=y_bottom_arrays
            ),
            "correlation_matrix_concentration": self.correlation_matrix_concentration,
            "noise_scale": self.noise_scale,
        }

        return dict(
            t=t_scaled,
            y=y_bottom_arrays,
            data=exogenous_data,
            changepoint_matrix=changepoint_matrix,
            **self.fit_and_predict_data_,
        )

    def _get_exogenous_matrix_from_X(self, X: pd.DataFrame) -> jnp.ndarray:
        """
        Convert the exogenous variables to a NumPyro matrix.

        Args:
            X (pd.DataFrame): The exogenous variables.

        Returns:
            jnp.ndarray: The NumPyro matrix of the exogenous variables.
        """
        X_bottom = loc_bottom_series(X)
        X_arrays = series_to_tensor(X_bottom)

        return X_arrays

    def _set_time_scale(self, t_arrays):
        """
        Setup model parameters and scalers.

        This function has the collateral effect of setting the following attributes:
        - self._time_scaler
        - self.max_t
        - self.min_t


        Args:
            t_arrays (ndarray): Transformed time index.

        Returns:
            None
        """

        # Scale time index
        self._time_scaler = TimeScaler()
        self._time_scaler.fit(t=t_arrays)
        t_scaled = self._time_scaler.scale(t=t_arrays)

        # Setting loc and scales for the priors
        self.max_t = t_scaled.max(axis=1).squeeze()
        self.min_t = t_scaled.min(axis=1).squeeze()

    @property
    def _capacity_prior_loc(self):
        """
        The prior location of the capacity parameter.

        Returns:
            float: The prior location of the capacity parameter.
        """

        val = self.capacity_prior_loc
        if isinstance(val, Iterable):
            return jnp.array(val)
        return jnp.ones(self.n_series) * val

    @property
    def _capacity_prior_scale(self):
        """
        The capacity prior scale

        Returns a float or an array-like with the same shape as the number of series.
        """
        val = self.capacity_prior_scale
        if isinstance(val, Iterable):
            return jnp.array(val)
        return jnp.ones(self.n_series) * val

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

    def _get_changepoint_matrix(self, t_scaled):
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

    def _get_changepoint_prior_vectors(
        self,
        n_changepoint_per_series: List[int],
        changepoint_prior_scale: float,
        global_rates: jnp.array,
    ):
        """

        Returns the prior vectors for the changepoint coefficients.

        Args:
            n_changepoint_per_series (List[int]): Number of changepoints for each series.
            changepoint_prior_scale (float): Scale parameter for the changepoint prior.
            global_rates (jnp.array): Global rates for each series.

        Returns:
            None
        """

        n_series = len(n_changepoint_per_series)

        def zeros_with_first_value(size, first_value):
            x = jnp.zeros(size)
            x.at[0].set(first_value)
            return x

        changepoint_prior_scale_vector = np.concatenate(
            [
                np.ones(n_changepoint) * cur_changepoint_prior_scale
                for n_changepoint, cur_changepoint_prior_scale in zip(
                    n_changepoint_per_series,
                    to_list_if_scalar(changepoint_prior_scale, n_series),
                )
            ]
        )

        changepoint_prior_loc_vector = np.concatenate(
            [
                zeros_with_first_value(n_changepoint, estimated_global_rate)
                for n_changepoint, estimated_global_rate in zip(
                    n_changepoint_per_series, global_rates
                )
            ]
        )

        return jnp.array(changepoint_prior_loc_vector), jnp.array(
            changepoint_prior_scale_vector
        )

    def _get_trend_sample_func(
        self, t_arrays: jnp.ndarray, y_arrays: jnp.ndarray
    ) -> Callable:
        """

        Get a function that samples the trend parameters.

        This function may change in the future. Currently, the model function receives a function to get the changepoint coefficients.
        This function is passed to the model as a partial function, with the changepoint coefficients as a parameter

        Args:
            t_arrays (jnp.array): array with shape (n_timepoints, 1).
            y_arrays (jnp.array): array with shape (n_series, n_timepoints, 1).

        Returns:
            Callable: Function that samples the trend parameters.

        """

        distributions = self._get_trend_prior_distributions(t_arrays, y_arrays)

        def trend_sample_func(distributions):

            return init_params(distributions)

        return functools.partial(trend_sample_func, distributions=distributions)

    def _get_trend_prior_distributions(self, t_arrays, y_arrays):

        distributions = {}

        if self.trend == "linear":
            global_rates = enforce_array_if_zero_dim(
                (y_arrays[:, -1].squeeze() - y_arrays[:, 0].squeeze())
                / (t_arrays[0].squeeze() - t_arrays[-1].squeeze())
            )
            offset = y_arrays[:, 0].squeeze() - global_rates * t_arrays[0].squeeze()

            distributions["offset"] = dist.Normal(
                offset,
                jnp.array(self.changepoint_prior_scale) * 10,
            )

        if self.trend == "logistic":

            global_rates, offset = suggest_logistic_rate_and_offset(
                t=t_arrays.squeeze(),
                y=y_arrays.squeeze(),
                capacities=self._capacity_prior_loc,
            )
            # We subtract one because it is later added to capacity (inside the model)
            distributions["capacity"] = dist.TransformedDistribution(
                dist.HalfNormal(
                    self._capacity_prior_scale,
                ),
                dist.transforms.AffineTransform(
                    loc=self._capacity_prior_loc, scale=1
                ),
            )

            distributions["offset"] = dist.Normal(
                offset,
                jnp.array(self.changepoint_prior_scale) * 10,
            )

        # Trend

        # Changepoints and trend-related distributions
        changepoint_prior_loc_vector, changepoint_prior_scale_vector = (
            self._get_changepoint_prior_vectors(
                n_changepoint_per_series=self.n_changepoint_per_series,
                changepoint_prior_scale=self.changepoint_prior_scale,
                global_rates=global_rates,
            )
        )

        distributions["changepoint_coefficients"] = dist.Laplace(
            changepoint_prior_loc_vector, changepoint_prior_scale_vector
        )

        return distributions

    def predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """
        samples = super().predict_samples(X=X, fh=fh)

        return self.aggregator_.transform(samples)

    

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """
        
        fh_dates = fh.to_absolute(
            cutoff=self.full_y_indexes_.get_level_values(-1).max()
        )
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        t_arrays = jnp.array(convert_index_to_days_since_epoch(fh_as_index)).reshape(
            (1, -1, 1)
        )
        t_arrays = jnp.tile(t_arrays, (self.n_series, 1, 1))
        t_arrays = self._time_scaler.scale(t_arrays)
        t_scaled = t_arrays[0]

        changepoint_matrix = self._get_changepoint_matrix(t_scaled)

        if self._has_exogenous_variables:
            if X is None or X.shape[1] == 0:
                idx = reindex_time_series(self._y, fh_as_index).index
                X = pd.DataFrame(index=idx)
                X = self.aggregator_.transform(X)

            X = X.loc[X.index.get_level_values(-1).isin(fh_as_index)]
            if self.feature_transformer is not None:
                X = self.feature_transformer.transform(X)
            X = self.expand_columns_transformer_.transform(X)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))
        else:
            exogenous_data = {}

        return dict(
            t=t_arrays,
            y=None,
            data=exogenous_data,
            changepoint_matrix=changepoint_matrix,
            **self.fit_and_predict_data_,
        )

    def periodindex_to_multiindex(self, periodindex: pd.PeriodIndex) -> pd.MultiIndex:
        """
        Convert a PeriodIndex to a MultiIndex.

        Args:
            periodindex (pd.PeriodIndex): PeriodIndex to convert.

        Returns:
            pd.MultiIndex: Converted MultiIndex.
        """
        if self._y.index.nlevels == 1:
            return periodindex

        levels = self._y.index.droplevel(-1).unique().tolist()
        # import Iterable
        from collections.abc import Iterable

        # Check if base_levels 0 is a iterable, because, if there are more than
        # one level, the objects are tuples. If there's only one level, the
        # objects inside levels list are strings. We do that to make it more "uniform"
        if not isinstance(levels[0], tuple):
            levels = [(x,) for x in levels]

        bottom_levels = [idx for idx in levels if idx[-1] != "__total"]

        return pd.Index(
            map(
                lambda x: (*x[0], x[1]),
                itertools.product(bottom_levels, periodindex),
            ),
            name=self._y.index.names,
        )

    @property
    def n_changepoint_per_series(self):
        """Get the number of changepoints per series.

        Returns:
            int: Number of changepoints per series.
        """
        return [len(cp) for cp in self._changepoint_ts]

    @property
    def n_series(self):
        """Get the number of series.

        Returns:
            int: Number of series.
        """
        return self.hierarchy_matrix.shape[1]


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


class TimeScaler:

    def fit(self, t):
        """
        Fit the time scaler.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            TimeScaler: The fitted TimeScaler object.
        """

        if t.ndim == 1:
            t = t.reshape(1, -1)
        self.t_scale = (t[:, 1:] - t[:, :-1]).flatten().mean()
        self.t_min = t.min()
        return self

    def scale(self, t):
        """
        Transform the time indices.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Transformed time indices.
        """
        return (t - self.t_min) / self.t_scale

    def fit_scale(self, t):
        """
        Fit the time scaler and transform the time indices.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Transformed time indices.
        """
        self.fit(t)
        return self.scale(t)
