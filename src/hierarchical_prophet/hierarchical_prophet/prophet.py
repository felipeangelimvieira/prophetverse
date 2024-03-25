import itertools
import logging
import re
from functools import partial
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

from hierarchical_prophet.hierarchical_prophet._changepoint_matrix import ChangepointMatrix

from hierarchical_prophet._utils import (
    convert_dataframe_to_tensors,
    convert_index_to_days_since_epoch,
    extract_timetensor_from_dataframe,
    get_multiindex_loc,
    loc_bottom_series,
    series_to_tensor,
)
from hierarchical_prophet.multiindex import reindex_time_series
from hierarchical_prophet.logistic import suggest_logistic_rate_and_offset
from hierarchical_prophet.base import BaseBayesianForecaster, init_params
from hierarchical_prophet.exogenous_priors import (
    get_exogenous_priors,
    sample_exogenous_coefficients,
)
from hierarchical_prophet.hierarchical_prophet._distribution import NormalReconciled
from hierarchical_prophet.hierarchical_prophet._expand_column_per_level import (
    ExpandColumnPerLevel,
)
from hierarchical_prophet.jax_functions import (
    additive_mean_model,
    multiplicative_mean_model,
)

logger = logging.getLogger("sktime-numpyro")


class HierarchicalProphet(BaseBayesianForecaster):
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
        exogenous_priors={},
        default_exogenous_prior=(dist.Normal, 0, 1),
        changepoint_prior_scale=0.1,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        y_scales=None,
        noise_scale=0.05,
        trend="linear",
        seasonality_mode="multiplicative",
        transformer_pipeline: BaseTransformer = None,
        individual_regressors=[],
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
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
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.noise_scale = noise_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.y_scales = y_scales
        self.seasonality_mode = seasonality_mode
        self.trend = trend
        self.exogenous_priors = exogenous_priors
        self.default_exogenous_prior = default_exogenous_prior
        self.individual_regressors = individual_regressors
        self.transformer_pipeline = transformer_pipeline

        super().__init__(
            rng_key=rng_key,
            method="mcmc",
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
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
        if self.changepoint_range is not None and (self.changepoint_range <= 0 or self.changepoint_range > 1):
            raise ValueError("changepoint_range must be in the range (0, 1].")
        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be greater than 0.")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")
        if self.capacity_prior_scale <= 0:
            raise ValueError("capacity_prior_scale must be greater than 0.")
        if self.capacity_prior_loc <= 0:
            raise ValueError("capacity_prior_loc must be greater than 0.")
        if self.y_scales is not None:
            if any([scale <= 0 for scale in self.y_scales]):
                raise ValueError("y_scales must be greater than 0.")
        if self.seasonality_mode not in ["multiplicative", "additive"]:
            raise ValueError('seasonality_mode must be either "multiplicative" or "additive".')
        if self.trend not in ["linear", "logistic"]:
            raise ValueError('trend must be either "linear" or "logistic".')

        if y is not None:
            if self.y_scales is not None:
                if len(self.y_scales) != len(y.columns):
                    raise ValueError("y_scales must have the same length as the number of columns in y.")

    def _get_numpyro_model_data(self, y, X, fh):
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

        # Changepoints
        self._setup_changepoint_matrix_transformer(t_scaled)
        changepoints_matrix, changepoints_mask = (
            self._changepoint_matrix_maker.transform(t=t_scaled)
        )

        # Exog variables
        self.exogenous_columns_ = set([])
        if self._has_exogenous_variables:

            self.expand_columns_transformer_ = ExpandColumnPerLevel(
                self.individual_regressors
            ).fit(X)
            X = X.loc[y.index]
            X = self.expand_columns_transformer_.transform(X)
            exogenous_matrix = self._get_exogenous_matrix_from_X(X)
            self.exogenous_columns_ = X.columns

        else:
            exogenous_matrix = jnp.zeros_like(t_scaled)

        self.extra_inputs_ = {
            "s_matrix": self.hierarchy_matrix,
            "changepoints_matrix": changepoints_matrix,
            "changepoints_mask": changepoints_mask,
            "y_scales" : self.y_scales,
            "seasonality_mode" : self.seasonality_mode,
            "trend_mode" : self.trend,
            **self._set_prior_distributions(t_scaled, y_bottom_arrays, X),
        }

        return dict(
            t=t_scaled, y=y_arrays.squeeze().T, X=exogenous_matrix, **self.extra_inputs_
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
        if self.y_scales is None:
            self.y_scales = np.abs(y_arrays).max(axis=1).squeeze()

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

    def _setup_changepoint_matrix_transformer(self, t_scaled):
        """
        Setup changepoint variables and transformer.

        Args:
            t_arrays (ndarray): Transformed time index.

        Returns:
            None
        """
        self._changepoint_matrix_maker = ChangepointMatrix(
            to_list_if_scalar(self.changepoint_interval, self.n_series),
            changepoint_range=to_list_if_scalar(
                self.changepoint_range or -self.changepoint_interval, self.n_series
            ),
        )
        self._changepoint_matrix_maker.fit(t=t_scaled)

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

    def _get_trend_prior_distributions(self, t_arrays, y_arrays):

        distributions = {}

        if self.trend == "linear":
            global_rates = enforce_array_if_zero_dim(
                (self.max_y - self.min_y) / (self.max_t - self.min_t) / self.y_scales
            )
            offset = (self.min_y - global_rates * self.min_t) / self.y_scales

            distributions["offset"] = dist.Normal(
                offset,                
                jnp.array(self.changepoint_prior_scale)*10,
            )

        if self.trend == "logistic":

            global_rates, offset = suggest_logistic_rate_and_offset(
                t=t_arrays.squeeze(),
                y=y_arrays.squeeze(),
                capacities=self._capacity_prior_loc * self.y_scales,
            )
            global_rates /= self.y_scales

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

        distributions["changepoints_coefficients"] = dist.Laplace(
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

            distributions["exogenous_coefficients"] = OrderedDict(exogenous_distributions)
            inputs["exogenous_permutation_matrix"] = exogenous_permutation_matrix
        else:
            inputs["exogenous_permutation_matrix"] = None
            # distributions["exogenous_coefficients"] = None

        # Noise
        distributions["std_observation"] = dist.HalfNormal(
            self.noise_scale * self.y_scales
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

        return get_multiindex_loc(
            samples, self.original_y_indexes_.droplevel(-1).unique()
        )

    def _predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
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

        changepoints_matrix, changepoints_mask = (
            self._changepoint_matrix_maker.transform(t=t_arrays)
        )

        if self._has_exogenous_variables:
            X = X.loc[X.index.get_level_values(-1).isin(fh_as_index)]
            X = self.expand_columns_transformer_.transform(X)
            exogenous_matrix = self._get_exogenous_matrix_from_X(X)
        else:
            exogenous_matrix = jnp.zeros_like(t_arrays)

        predictive = Predictive(
            self.model,
            self.posterior_samples_,
            return_sites=["obs", *self.site_names],
        )

        extra_inputs_ = self.extra_inputs_.copy()
        extra_inputs_.update(
            {
                "changepoints_matrix": changepoints_matrix,
                "changepoints_mask": changepoints_mask,
            }
        )

        return predictive(
            self.rng_key, t=t_arrays, y=None, X=exogenous_matrix, **extra_inputs_
        )

    def _set_exogenous_priors(self, X):
        """Set the prior distributions for the exogenous variables.

        Args:
            X (pd.DataFrame): Exogenous variables.

        Returns:
            self
        """
        # The dict self.exogenous_prior contain a regex as key, and a tuple (distribution, kwargs) as value.
        # The regex is matched against the column names of X, and the corresponding distribution is used to

        (
            self._exogenous_dists,
            self._exogenous_permutation_matrix,
        ) = get_exogenous_priors(X, self.exogenous_priors, self.default_exogenous_prior)
        return self

    @property
    def n_changepoint_per_series(self):
        """Get the number of changepoints per series.

        Returns:
            int: Number of changepoints per series.
        """
        return self._changepoint_matrix_maker.n_changepoint_per_series

    @property
    def n_series(self):
        """Get the number of series.

        Returns:
            int: Number of series.
        """
        return self.hierarchy_matrix.shape[1]


def model(
    y,
    X,
    changepoints_matrix,
    changepoints_mask,
    s_matrix,
    exogenous_permutation_matrix,
    distributions,
    trend_mode: str,
    seasonality_mode: str,
    y_scales: str,
    *args,
    **kwargs,
) -> None:
    """
    Define the probabilistic model for Prophet2.

    Args:
        t (ndarray): Time index.
        y (ndarray): Target time series.
        X (ndarray): Exogenous variables.
        changepoints_matrix (ndarray): Changepoint matrix.
        s_matrix (ndarray): Reconciliation matrix.
        args, kwargs: Additional arguments and keyword arguments for the model.

    Returns:
        None
    """
    params = init_params(distributions=distributions)

    # Trend

    changepoints_coefficients = params["changepoints_coefficients"]
    offset = params["offset"]

    trend = (
        changepoints_matrix * (changepoints_mask * changepoints_coefficients.flatten())
    ).sum(axis=-1)
    trend = jnp.expand_dims(trend, axis=-1) + offset.reshape((-1, 1, 1))

    if trend_mode == "linear":
        trend *= y_scales.reshape((-1, 1, 1))

    elif trend_mode == "logistic":
        capacity = params["capacity"]
        capacity = capacity.reshape((-1, 1, 1)) * y_scales.reshape((-1, 1, 1))
        trend = capacity / (1 + jnp.exp(-trend))

    numpyro.deterministic("trend_", trend)

    # Exogenous variables

    if params.get("exogenous_coefficients") is not None:
        exogenous_coefficients = params["exogenous_coefficients"]
        permuted_exogenous = exogenous_permutation_matrix @ exogenous_coefficients

        exogenous_effect_decomposition = X * permuted_exogenous.reshape((1, -1))
        numpyro.deterministic(
            "exogenous_effect_decomposition_", exogenous_effect_decomposition
        )
        exogenous_effect = exogenous_effect_decomposition.sum(axis=-1).reshape(
            (X.shape[0], X.shape[1], 1)
        )
        if seasonality_mode == "additive":
            exogenous_effect = exogenous_effect * y_scales.reshape((-1, 1, 1))

        # Mean
        if seasonality_mode == "additive":
            mean = additive_mean_model(trend, exogenous_effect)

        elif seasonality_mode == "multiplicative":
            mean = multiplicative_mean_model(trend, exogenous_effect, exponent=1)
        mean = mean.squeeze()
    else:
        mean = trend.squeeze()

    numpyro.deterministic("mean_", mean)

    if mean.ndim == 1:
        mean = mean.reshape((1, -1))


    numpyro.sample(
        "obs",
        NormalReconciled(
            mean.T,
            params["std_observation"],
            reconc_matrix=s_matrix,
        ),
        obs=y,
    )


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
