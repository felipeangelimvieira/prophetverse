import itertools
import logging
import re
import functools
from typing import Dict, List, Tuple, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import lax, random
from collections import OrderedDict
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import _get_s_matrix

from hierarchical_prophet.hierarchical_prophet._time_scaler import TimeScaler

from hierarchical_prophet.hierarchical_prophet._changepoint_matrix import (
    ChangepointMatrix,
)

from hierarchical_prophet.utils.frame_to_array import (
    convert_dataframe_to_tensors,
    convert_index_to_days_since_epoch,
    extract_timetensor_from_dataframe,
    get_multiindex_loc,
    loc_bottom_series,
    series_to_tensor,
)
from hierarchical_prophet.utils.multiindex import reindex_time_series
from hierarchical_prophet.utils.logistic import suggest_logistic_rate_and_offset
from hierarchical_prophet.sktime.base import BaseBayesianForecaster, ExogenousEffectMixin, init_params
from hierarchical_prophet.utils.exogenous_priors import (
    get_exogenous_priors,
    sample_exogenous_coefficients,
)
from hierarchical_prophet.hierarchical_prophet._distribution import NormalReconciled
from hierarchical_prophet.hierarchical_prophet._expand_column_per_level import (
    ExpandColumnPerLevel,
)
from hierarchical_prophet.utils.jax_functions import (
    additive_mean_model,
    multiplicative_mean_model,
)
from hierarchical_prophet.trend_utils import (
    get_changepoint_matrix,
    get_changepoint_timeindexes,
)

from hierarchical_prophet.multivariate.model import model
from hierarchical_prophet.effects import LinearEffect, LinearHeterogenousPriorsEffect

logger = logging.getLogger("sktime-numpyro")


class HierarchicalProphet(ExogenousEffectMixin, BaseBayesianForecaster):
    """A class that represents a Bayesian hierarchical time series forecasting model based on the Prophet algorithm.

    Args:
        changepoint_interval (int, optional): The interval between potential changepoints in the time series. Defaults to 25.
        changepoint_range (float or None, optional): The index of the last changepoint. If None, default to -changepoint_freq. Note that this is the index in the list of timestamps, and because the timeseries increases in size, it is better to use negative indexes instead of positive. Defaults to None.
        exogenous_priors (dict, optional): A dictionary of prior distributions for the exogenous variables. The keys are regexes, or the name of an specific column, and the values are tuples of the format (dist.Distribution, *args). The args are passed to dist.Distributions at the moment of sampling. Defaults to {}.
        default_exogenous_prior (tuple, optional): The default prior distribution for the exogenous variables. Defaults to (dist.Normal, 0, 1), and is applied to columns not included in exogenous_priors above.
        changepoint_prior_scale (float, optional): The scale parameter for the prior distribution of changepoints. Defaults to 0.1. Note that this parameter is scaled by the magnitude of the timeseries.
        capacity_prior_loc (float, optional): The location parameter for the prior distribution of capacity. This parameter is scaled according to y_scales. Defaults to 1.1, i.e., 110% of the maximum value of the series, or the y_scales if provided.
        capacity_prior_scale (float, optional): The scale parameter for the prior distribution of capacity. This parameter is scaled according to y_scales. Defaults to 0.2.
        y_scales (array-like or None, optional): The scales of the target time series. Defaults to None. If not provided, the scales are computed from the data, according to its maximum value.
        noise_scale (float, optional): The scale parameter for the prior distribution of observation noise. Defaults to 0.05.
        trend (str, optional): The type of trend to model. Possible values are "linear" and "logistic". Defaults to "linear".
        seasonality_mode (str, optional): The mode of seasonality to model. Possible values are "multiplicative" and "additive". Defaults to "multiplicative".
        transformer_pipeline (BaseTransformer): A BaseTransformer or TransformerPipeline from sktime to apply to X and create features internally, such as Fourier series. See sktime's Transformer documentation for more information. Defaults to None.
        individual_regressors (list, optional): A list of regexes/names of individual regressors to separate by timeseries. If not provided, all regressors are shared. Defaults to [].
        mcmc_samples (int, optional): The number of MCMC samples to draw. Defaults to 2000.
        mcmc_warmup (int, optional): The number of MCMC warmup steps. Defaults to 200.
        mcmc_chains (int, optional): The number of MCMC chains to run in parallel. Defaults to 4.
        rng_key (jax.random.PRNGKey, optional): The random number generator key. Defaults to random.PRNGKey(24).


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
        
        default_exogenous_prior=("Normal", 0, 1),
        exogenous_effects=None,
        changepoint_prior_scale=0.1,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="linear",
        seasonality_mode="multiplicative",
        transformer_pipeline: BaseTransformer = None,
        individual_regressors=[],
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs={"step_size" : 1e-4},
        optimizer_steps=100_000,
        correlation_matrix_concentration=1.0,
        rng_key=random.PRNGKey(24),
    ):
        """
        Initialize the HierarchicalProphet forecaster.

        Args:
            changepoint_interval (int, optional): The interval between potential changepoints in the time series. Defaults to 25.
            changepoint_range (float or None, optional): The index of the last changepoint. If None, default to -changepoint_freq. Note that this is the index in the list of timestamps, and because the timeseries increases in size, it is better to use negative indexes instead of positive. Defaults to None.
            exogenous_priors (dict, optional): A dictionary of prior distributions for the exogenous variables. The keys are regexes, or the name of an specific column, and the values are tuples of the format (dist.Distribution, *args). The args are passed to dist.Distributions at the moment of sampling. Defaults to {}.
            default_exogenous_prior (tuple, optional): The default prior distribution for the exogenous variables. Defaults to (dist.Normal, 0, 1), and is applied to columns not included in exogenous_priors above.
            changepoint_prior_scale (float, optional): The scale parameter for the prior distribution of changepoints. Defaults to 0.1. Note that this parameter is scaled by the magnitude of the timeseries.
            capacity_prior_loc (float, optional): The location parameter for the prior distribution of capacity. This parameter is scaled according to y_scales. Defaults to 1.1, i.e., 110% of the maximum value of the series, or the y_scales if provided.
            capacity_prior_scale (float, optional): The scale parameter for the prior distribution of capacity. This parameter is scaled according to y_scales. Defaults to 0.2.
            
            noise_scale (float, optional): The scale parameter for the prior distribution of observation noise. Defaults to 0.05.
            trend (str, optional): The type of trend to model. Possible values are "linear" and "logistic". Defaults to "linear".
            seasonality_mode (str, optional): The mode of seasonality to model. Possible values are "multiplicative" and "additive". Defaults to "multiplicative".
            transformer_pipeline (BaseTransformer): A BaseTransformer or TransformerPipeline from sktime to apply to X and create features internally, such as Fourier series. See sktime's Transformer documentation for more information. Defaults to None.
            individual_regressors (list, optional): A list of regexes/names of individual regressors to separate by timeseries. If not provided, all regressors are shared. Defaults to [].
            mcmc_samples (int, optional): The number of MCMC samples to draw. Defaults to 2000.
            mcmc_warmup (int, optional): The number of MCMC warmup steps. Defaults to 200.
            mcmc_chains (int, optional): The number of MCMC chains to run in parallel. Defaults to 4.
            rng_key (jax.random.PRNGKey, optional): The random number generator key. Defaults to random.PRNGKey(24).
        """
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.noise_scale = noise_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.seasonality_mode = seasonality_mode
        self.trend = trend
        self.default_exogenous_prior = default_exogenous_prior
        self.individual_regressors = individual_regressors
        self.transformer_pipeline = transformer_pipeline
        self.correlation_matrix_concentration = correlation_matrix_concentration

        super().__init__(
            rng_key=rng_key,
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

        self.aggregator_ = None
        self.original_y_indexes_ = None
        self.full_y_indexes_ = None
        self.hierarchy_matrix = None
        self._has_exogenous_variables = None
        self.expand_columns_transformer_ = None
        self._time_scaler = None
        self.max_y = None
        self.min_y = None
        self.max_t = None
        self.min_t = None
        self._changepoint_matrix_maker = None

        self._validate_hyperparams()

    def _validate_hyperparams(self, y=None):
        """
        Validate the hyperparameters of the HierarchicalProphet forecaster.
        """
        if self.changepoint_interval <= 0:
            raise ValueError("changepoint_interval must be greater than 0.")
        # if self.changepoint_range is not None and (
        #    self.changepoint_range <= 0 or self.changepoint_range > 1
        # ):
        #    raise ValueError("changepoint_range must be in the range (0, 1].")
        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be greater than 0.")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")
        if self.capacity_prior_scale <= 0:
            raise ValueError("capacity_prior_scale must be greater than 0.")
        if self.capacity_prior_loc <= 0:
            raise ValueError("capacity_prior_loc must be greater than 0.")
        if self.seasonality_mode not in ["multiplicative", "additive"]:
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

        self._validate_hyperparams(y=y)
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

        # If no exogenous variables, create empty DataFrame
        # Else, aggregate exogenous variables and transform them
        if X is None or X.columns.empty:

            X = pd.DataFrame(index=y.index)
            if self.transformer_pipeline is None:
                self._has_exogenous_variables = False
            else:
                self._has_exogenous_variables = True

        else:
            self._has_exogenous_variables = True
            X = self.aggregator_.transform(X)

        if self.transformer_pipeline is not None:
            X = self.transformer_pipeline.fit_transform(X)

        y_bottom = loc_bottom_series(y)

        # Convert inputs to array, including the time index
        y_arrays = series_to_tensor(y)
        y_bottom_arrays = series_to_tensor(y_bottom)
        t_arrays = extract_timetensor_from_dataframe(y_bottom)

        # Setup model parameters and scalers, and get
        self._setup_scales(t_arrays, y_bottom_arrays)

        t_scaled = self._time_scaler.scale(t_arrays)
        t_scaled = t_scaled[0].reshape((-1, 1))
        # Changepoints
        self._setup_changepoints(t_scaled=t_scaled)
        changepoint_matrix = self._get_changepoint_matrix(t_scaled)

        # Exog variables
        self.exogenous_columns_ = set([])
        if self._has_exogenous_variables:

            self.expand_columns_transformer_ = ExpandColumnPerLevel(
                self.individual_regressors
            ).fit(X)
            X = X.loc[y.index]
            X = self.expand_columns_transformer_.transform(X)

            self._set_custom_effects(feature_names=X.columns)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))

        else:
            exogenous_data = {}

        self.extra_inputs_ = {
            "changepoint_matrix": changepoint_matrix,
            "trend_mode": self.trend,
            "exogenous_effects": self.exogenous_effect_dict,
            "init_trend_params": self._get_trend_sample_func(
                t_arrays=t_scaled, y_arrays=y_bottom_arrays
            ),
            "correlation_matrix_concentration": self.correlation_matrix_concentration,
            "noise_scale" : self.noise_scale,
        }

        return dict(
            t=t_scaled,
            y=y_bottom_arrays,
            data=exogenous_data,
            **self.extra_inputs_,
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

    def _setup_scales(self, t_arrays, y_arrays):
        """
        Setup model parameters and scalers.

        Args:
            t_arrays (ndarray): Transformed time index.
            y_arrays (ndarray): Transformed target time series.

        Returns:
            None
        """

        # Scale time index
        self._time_scaler = TimeScaler()
        self._time_scaler.fit(t=t_arrays)
        t_scaled = self._time_scaler.scale(t=t_arrays)

        # Setting loc and scales for the priors
        self.max_y = y_arrays.max(axis=1).squeeze()
        self.min_y = y_arrays.min(axis=1).squeeze()
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
        val = self.capacity_prior_scale
        if isinstance(val, Iterable):
            return jnp.array(val)
        return jnp.ones(self.n_series) * val

    def _setup_changepoints(self, t_scaled):
        """
        Setup changepoint variables and transformer.

        Args:
            t_arrays (ndarray): Transformed time index.

        Returns:
            None
        """
        changepoint_intervals = (
            to_list_if_scalar(self.changepoint_interval, self.n_series)
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
        Get the changepoint matrix.

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

    def get_changepoint_prior_vectors(
        self, n_changepoint_per_series, changepoint_prior_scale, n_series, global_rates
    ):
        """
        Set the prior vectors for changepoint distribution.

        Returns:
            None
        """

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

    def _get_trend_sample_func(self, t_arrays, y_arrays):

        distributions = self._get_trend_prior_distributions(t_arrays, y_arrays)

        def trend_sample_func(distributions):

            return init_params(distributions)

        return functools.partial(trend_sample_func, distributions=distributions)

    def _get_trend_prior_distributions(self, t_arrays, y_arrays):

        distributions = {}

        if self.trend == "linear":
            global_rates = enforce_array_if_zero_dim(
                (self.max_y - self.min_y) / (self.max_t - self.min_t) 
            )
            offset = (self.min_y - global_rates * self.min_t) 

            distributions["offset"] = dist.Normal(
                offset,
                jnp.array(self.changepoint_prior_scale) * 10,
            )

        if self.trend == "logistic":

            global_rates, offset = suggest_logistic_rate_and_offset(
                t=t_arrays.squeeze(),
                y=y_arrays.squeeze(),
                capacities=self._capacity_prior_loc ,
            )

            distributions["capacity"] = dist.LogNormal(
                self._capacity_prior_loc,
                self._capacity_prior_scale,
            )

            distributions["offset"] = dist.Normal(
                offset,
                jnp.log(
                    y_arrays.max(axis=1).squeeze().flatten()
                    - y_arrays.min(axis=1).squeeze().flatten()
                ),
            )

        # Trend

        # Changepoints and trend-related distributions
        changepoint_prior_loc_vector, changepoint_prior_scale_vector = (
            self.get_changepoint_prior_vectors(
                n_changepoint_per_series=self.n_changepoint_per_series,
                changepoint_prior_scale=self.changepoint_prior_scale,
                n_series=self.n_series,
                global_rates=global_rates,
            )
        )

        distributions["changepoint_coefficients"] = dist.Laplace(
            changepoint_prior_loc_vector, changepoint_prior_scale_vector
        )

        return distributions

    def _set_prior_distributions(
        self, t_arrays: jnp.ndarray, y_arrays: jnp.ndarray, X: pd.DataFrame
    ):
        """Set the prior distributions.

        Args:
            t_arrays (jnp.ndarray): The array of time values.
            y_arrays (jnp.ndarray): The array of target values.
            X (pd.DataFrame): The dataframe of exogenous variables.

        Returns:
            None
        """

        distributions = {}
        inputs = {}

        # Trend
        distributions.update(self._get_trend_prior_distributions(t_arrays, y_arrays))

        # Exog variables
        exogenous_distributions = None
        if self._has_exogenous_variables:
            # The dict self.exogenous_prior contain a regex as key, and a tuple (distribution, kwargs) as value.
            # The regex is matched against the column names of X, and the corresponding distribution is used to

            exogenous_distributions, exogenous_permutation_matrix = (
                get_exogenous_priors(
                    X, self.exogenous_priors, self.default_exogenous_prior
                )
            )

            distributions["exogenous_coefficients"] = OrderedDict(
                exogenous_distributions
            )
            inputs["exogenous_permutation_matrix"] = exogenous_permutation_matrix
        else:
            inputs["exogenous_permutation_matrix"] = None
            # distributions["exogenous_coefficients"] = None

        # Noise
        distributions["std_observation"] = dist.HalfNormal(
            self.noise_scale
        )

        inputs["distributions"] = distributions

        return inputs

    def predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """
        samples = super().predict_samples(X, fh)

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

        if X is not None and X.shape[1] == 0:
            X = None

        if self.transformer_pipeline is not None:
            # Create an empty X if the model has transformer and X is None
            if X is None:
                idx = reindex_time_series(self._y, fh_as_index).index
                X = pd.DataFrame(index=idx)
            X = self.transformer_pipeline.transform(X)

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        t_arrays = jnp.array(convert_index_to_days_since_epoch(fh_as_index)).reshape(
            (1, -1, 1)
        )
        t_arrays = jnp.tile(t_arrays, (self.n_series, 1, 1))
        t_arrays = self._time_scaler.scale(t_arrays)
        t_scaled = t_arrays[0]

        changepoints_matrix = self._get_changepoint_matrix(t_scaled)

        if self._has_exogenous_variables:
            X = X.loc[X.index.get_level_values(-1).isin(fh_as_index)]
            X = self.expand_columns_transformer_.transform(X)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))
        else:
            exogenous_data = {}

        extra_inputs_ = self.extra_inputs_.copy()
        extra_inputs_.update(
            {
                "changepoint_matrix": changepoints_matrix,
            }
        )

        return dict(t=t_arrays, y=None, data=exogenous_data, **extra_inputs_)

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
