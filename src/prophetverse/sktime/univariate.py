""" Univariate Prophet model

This module implements the Univariate Prophet model, similar to the one implemented in the `prophet` library.

"""

from typing import Callable
from functools import partial
from sktime.transformations.series.detrend import Detrender
import jax.numpy as jnp
import pandas as pd
from numpyro import distributions as dist
from jax import random
from sktime.forecasting.base import ForecastingHorizon

from prophetverse.utils.frame_to_array import convert_index_to_days_since_epoch
from prophetverse.sktime.base import (
    BaseBayesianForecaster,
    ExogenousEffectMixin, init_params,
)


from sktime.transformations.base import BaseTransformer
from prophetverse.utils.logistic import suggest_logistic_rate_and_offset

from prophetverse.models.univariate_model import model
from prophetverse.changepoint import (
    get_changepoint_matrix,
    get_changepoint_timeindexes,
)
import functools


__all__ = ["Prophet"]


class Prophet(ExogenousEffectMixin, BaseBayesianForecaster):
    """

    Univariate Prophet model, similar to the one implemented in the `prophet` library.

    The main difference between the mathematical model here and from the original one is the
    logistic trend. Here, another parametrization is considered, and the capacity is not
    passed as input, but inferred from the data.

    With respect to API, this one follows sktime convention where all hiperparameters are passed during
    __init__, and uses `changepoint_interval` instead of `n_changepoints` to set the changepoints. There's no
    weekly_seasonality/yearly_seasonality, but instead, the user can pass a `feature_transformer` that
    will be used to generate the fourier terms. The same for holidays.

    This model accepts configurations where each exogenous variable has a different function relating it to its additive
    effect on the time series. One can, for example, set different priors for a group of feature, or use a Hill function
    to model the effect of a feature.


    Args:
        changepoint_interval (int): The number of points between each potential changepoint.
        changepoint_range (float): Proportion of the history in which trend changepoints will be estimated. 
                                   If a float between 0 and 1, the range will be that proportion of the history. 
                                   If an int, the range will be that number of points. A negative int indicates number of points
                                   counting from the end of the history.
        changepoint_prior_scale (float): Parameter controlling the flexibility of the automatic changepoint selection.
        feature_transformer (BaseTransformer): Transformer object to generate Fourier terms, holiday or other features. Should be a sktime's Transformer
        capacity_prior_scale (float): Scale parameter for the prior distribution of the capacity.
        capacity_prior_loc (float): Location parameter for the prior distribution of the capacity.
        noise_scale (float): Scale parameter for the observation noise.
        trend (str): Type of trend to use. Can be "linear" or "logistic".
        mcmc_samples (int): Number of MCMC samples to draw.
        mcmc_warmup (int): Number of MCMC warmup steps.
        mcmc_chains (int): Number of MCMC chains to run in parallel.
        inference_method (str): Inference method to use. Can be "mcmc" or "map".
        optimizer_name (str): Name of the optimizer to use for variational inference.
        optimizer_kwargs (dict): Additional keyword arguments to pass to the optimizer.
        optimizer_steps (int): Number of optimization steps to perform for variational inference.
        exogenous_effects (dict): Dictionary specifying the exogenous effects and their modes.
        default_effect_mode (str): Default mode for exogenous effects.
        default_exogenous_prior (tuple): Default prior distribution for exogenous effects.
        rng_key (jax.random.PRNGKey): Random number generator key.


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
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="linear",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs={"step_size" : 1e-4},
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect_mode="multiplicative",
        default_exogenous_prior=("Normal", 0, 1),
        rng_key=random.PRNGKey(24),
    ):
        """
        Initializes the Prophet model.
        """

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
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

        self.model = model
        self._validate_hyperparams()

    def _validate_hyperparams(self):
        """
        Validate the hyperparameters
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
        Prepares the data for the Numpyro model.

        Args:
            y (pd.DataFrame): Time series data.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of data for the Numpyro model.
        """

        self._set_time_scale(y)

        ## Changepoints and trend
        self._set_changepoints_t(y)
        t = self._index_to_scaled_timearray(y.index)
        changepoint_matrix = self._get_changepoint_matrix(t)
        trend_sample_func = self._get_trend_sample_func(y=y, X=X)

        ## Exogenous features

        if X is None:
            X = pd.DataFrame(index=y.index)

        if self.feature_transformer is not None:

            X = self.feature_transformer.fit_transform(X)

        self._has_exogenous = ~X.columns.empty
        X = X.loc[y.index]

        self._set_custom_effects(X.columns)
        exogenous_data = self._get_exogenous_data_array(X)

        y_array = jnp.array(y.values.flatten()).reshape((-1, 1))

        ## Inputs that also are used in predict
        self.fit_and_predict_data_ = {
            "init_trend_params": trend_sample_func,
            "trend_mode": self.trend,
            "exogenous_effects": self.exogenous_effect_dict if self._has_exogenous else None,
            }

        inputs = {
            "t": self._index_to_scaled_timearray(y.index),
            "y": y_array,
            "data": exogenous_data,
            "changepoint_matrix": changepoint_matrix,
            **self.fit_and_predict_data_,
        }

        return inputs

    def _get_trend_sample_func(self, y: pd.DataFrame, X: pd.DataFrame) -> Callable :
        """
        
        Get a function that samples the trend parameters.
        
        This function may change in the future. Currently, the model function receives a function to get the changepoint coefficients.
        This function is passed to the model as a partial function, with the changepoint coefficients as a parameter
        
        Args:
            y (pd.DataFrame): Time series data.
            X (pd.DataFrame): Exogenous variables.
            
        Returns:
            Callable: Function that samples the trend parameters.
        
        """
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

            distributions["capacity"] = dist.TransformedDistribution(
                dist.HalfNormal(
                    self.capacity_prior_scale,
                ),
                dist.transforms.AffineTransform(loc=self.capacity_prior_loc, scale=1),
            )

        distributions["std_observation"] = dist.HalfNormal(self.noise_scale)

        def init_trend_params(distributions) -> dict:

            return init_params(distributions)

        return functools.partial(init_trend_params, distributions=distributions)

    def _set_time_scale(self, y: pd.DataFrame):
        """
        Sets the scales for the time series data.

        Args:
            y (pd.DataFrame): Time series data.
        """

        # Set time scale
        t_days = convert_index_to_days_since_epoch(y.index)

        self.t_scale = (t_days[1:] - t_days[:-1]).mean()
        self.t_start = t_days.min() / self.t_scale

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

    def _set_changepoints_t(self, y : pd.DataFrame) -> None:
        """
        Sets the array of changepoint times.
        
        This function has the colateral effect of setting the attribute `_changepoint_t` in the class.

        Args:
            y (pd.DataFrame): Time series data.
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
    ) -> dict:
        """
        Prepares the data for making predictions.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of data for the Numpyro model.
        """

        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        t = self._index_to_scaled_timearray(fh_as_index)
        changepoint_matrix = self._get_changepoint_matrix(t)

        if X is None:
            X = pd.DataFrame(index=fh_as_index)

        if self.feature_transformer is not None:
            X = self.feature_transformer.transform(X)

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

   