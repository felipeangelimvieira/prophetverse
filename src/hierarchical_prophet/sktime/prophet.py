import logging
import re
from functools import partial
from collections import OrderedDict
from sktime.transformations.series.detrend import Detrender
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import lax, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sktime.forecasting.base import ForecastingHorizon

from hierarchical_prophet.utils.frame_to_array import convert_index_to_days_since_epoch
from hierarchical_prophet.sktime.base import (
    BaseBayesianForecaster,
    ExogenousEffectMixin, init_params,
)


from hierarchical_prophet.utils.logistic import suggest_logistic_rate_and_offset

from hierarchical_prophet.univariate.model import model
from hierarchical_prophet.changepoint import (
    get_changepoint_matrix,
    get_changepoint_timeindexes,
)
from hierarchical_prophet.effects import LinearHeterogenousPriorsEffect, LinearEffect
import functools


logger = logging.getLogger("hierarchical-prophet")

NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000


class Prophet(ExogenousEffectMixin, BaseBayesianForecaster):
    """
    
    
    """

    _tags = {
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        growth_offset_prior_scale=1,
        transformer_pipeline=None,
        capacity_prior_scale=None,
        capacity_prior_loc=1,
        noise_scale=0.05,
        trend="linear",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="mcmc",
        optimizer_name="Adam",
        optimizer_kwargs={},
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect_mode="multiplicative",
        default_exogenous_prior=("Normal", 0, 1),
        rng_key=random.PRNGKey(24),
    ):

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.noise_scale = noise_scale
        self.transformer_pipeline = transformer_pipeline
        self.growth_offset_prior_scale = growth_offset_prior_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend

        super().__init__(
            rng_key=rng_key,
            # ExogenousEffectMixin
            default_effect_mode=default_effect_mode,
            default_exogenous_prior=default_exogenous_prior,
            exogenous_effects=exogenous_effects,
            # BaseBayesianForecaster
            inference_method=inference_method,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
        )

        # Define all attributes that are created outside init

        self.t_start = None
        self.t_scale = None
        self.y_scale = None
        self._samples_predictive = None
        self.fourier_feature_transformer_ = None
        self.fit_and_predict_data_ = None
        self.exogenous_columns_ = None
        self.model = model

    def _get_fit_data(self, y, X, fh):
        """
        Prepares the data for the Numpyro model.

        Args:
            y (pd.DataFrame): Time series data.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of data for the Numpyro model.
        """

        self._set_time_and_y_scales(y)
        self._replace_hyperparam_nones_with_defaults(y)

        ## Changepoints
        self._set_changepoints_t(y)
        t = self._index_to_scaled_timearray(y.index)
        changepoint_matrix = self._get_changepoint_matrix(t)

        if self.transformer_pipeline is not None:
            if X is None:
                X = pd.DataFrame(index=y.index)
            X = self.transformer_pipeline.fit_transform(X)

        if X is None or X.columns.empty:
            self._has_exogenous = False
        else:
            self._has_exogenous = True
            X = X.loc[y.index]

        self._set_custom_effects(X.columns)
        exogenous_data = self._get_exogenous_data_array(X)

        y_array = jnp.array(y.values.flatten()).reshape((-1, 1))

        trend_sample_func = self._get_trend_sample_func(y=y, X=X)

        self.fit_and_predict_data_ = {
            "init_trend_params": trend_sample_func,
            "trend_mode": self.trend,
            "exogenous_effects": self.exogenous_effect_dict,
            }

        inputs = {
            "t": self._index_to_scaled_timearray(y.index),
            "y": y_array,
            "data": exogenous_data,
            "changepoint_matrix": changepoint_matrix,
            **self.fit_and_predict_data_,
        }

        return inputs

    def _get_trend_sample_func(self, y: pd.DataFrame, X: pd.DataFrame):
        t_scaled = self._index_to_scaled_timearray(y.index)
        distributions = {}

        changepoints_loc = jnp.zeros(len(self._changepoint_t))
        detrender = Detrender()
        trend = y - detrender.fit_transform(y)

        if self.trend == "linear":

            linear_global_rate = (trend.values[-1, 0] - trend.values[0, 0]) / (
                t_scaled[-1] - t_scaled[0]
            )
            changepoints_loc.at[0].set(linear_global_rate)

            distributions["changepoint_coefficients"] = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t)) * (self.changepoint_prior_scale),
            )

            distributions["offset"] = dist.Normal(
                (trend.values[0, 0] - linear_global_rate * t_scaled[0]),
                0.1,
            )

        if self.trend == "logistic":

            linear_global_rate, timeoffset = suggest_logistic_rate_and_offset(
                t_scaled,
                trend.values.flatten(),
                capacities=self.capacity_prior_loc,
            )

            linear_global_rate = linear_global_rate[0]
            timeoffset = timeoffset[0]
            changepoints_loc.at[0].set(linear_global_rate)

            changepoint_coefficients_distribution = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t)) * self.changepoint_prior_scale,
            )

            distributions["changepoint_coefficients"] = (
                changepoint_coefficients_distribution
            )

            distributions["offset"] = dist.Normal(timeoffset, jnp.log(2))

            distributions["capacity"] = dist.Normal(
                self.capacity_prior_loc,
                self.capacity_prior_scale,
            )

        distributions["std_observation"] = dist.HalfNormal(self.noise_scale)

        def init_trend_params(distributions) -> dict:

            return init_params(distributions)

        return functools.partial(init_trend_params, distributions=distributions)

    def _set_time_and_y_scales(self, y: pd.DataFrame):
        """
        Sets the scales for the time series data.

        Args:
            y (pd.DataFrame): Time series data.
        """

        # Set time scale
        t_days = convert_index_to_days_since_epoch(y.index)

        self.t_scale = (t_days[1:] - t_days[:-1]).mean()
        self.t_start = t_days.min() / self.t_scale

    def _replace_hyperparam_nones_with_defaults(self, y):
        """
        Replaces None values in hyperparameters with default values.

        Args:
            y (pd.DataFrame): Time series data.
        """
        if self.capacity_prior_loc is None:
            self.capacity_prior_loc = y.values.max() * 1.1
        if self.capacity_prior_scale is None:
            self.capacity_prior_scale = (y.values.max() - y.values.min()) * 0.2

    def _index_to_scaled_timearray(self, idx):
        """
        Scales the index values.

        Args:
            idx (pd.Index): Pandas Index object.

        Returns:
            np.ndarray: Scaled index values.
        """
        t_days = convert_index_to_days_since_epoch(idx)
        return (t_days) / self.t_scale - self.t_start

    def _convert_periodindex_to_floatarray(self, period_index):
        """
        Converts a pandas PeriodIndex object to a float array.

        Args:
            period_index (pd.PeriodIndex): Pandas PeriodIndex object.

        Returns:
            jnp.ndarray: Float array.
        """
        return jnp.array(self._index_to_scaled_timearray(period_index))

    def _set_changepoints_t(self, y):
        """
        Sets the array of changepoint times.

        Args:
            t (jnp.ndarray): Array of time values.
        """

        t_scaled = self._index_to_scaled_timearray(y.index)

        self._changepoint_t = get_changepoint_timeindexes(
            t=t_scaled,
            changepoint_interval=self.changepoint_interval,
            changepoint_range=self.changepoint_range,
        )

    def _get_changepoint_matrix(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Generates the changepoint coefficient matrix.

        Args:
            t (jnp.ndarray): Array of time values.

        Returns:
            jnp.ndarray: Changepoint coefficient matrix.
        """

        return get_changepoint_matrix(t, self._changepoint_t)

    def _get_predict_data(
        self, X: pd.DataFrame, fh: ForecastingHorizon
    ) -> pd.DataFrame:
        """
        Generates predictive samples.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of predictive samples.
        """
        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        t = self._index_to_scaled_timearray(fh_as_index)
        changepoint_matrix = self._get_changepoint_matrix(t)

        if self.transformer_pipeline is not None:
            if X is None:
                X = pd.DataFrame(index=fh_as_index)
            X = self.transformer_pipeline.fit_transform(X)


        exogenous_data = (
            self._get_exogenous_data_array(X.loc[fh_as_index]) if self._has_exogenous else None
        )

        return dict(
            t=t.reshape((-1, 1)),
            y=None,
            data=exogenous_data,
            changepoint_matrix=changepoint_matrix,
            **self.fit_and_predict_data_,
        )
