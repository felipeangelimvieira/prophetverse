""" Univariate Prophet model

This module implements the Univariate Prophet model, similar to the one implemented in the `prophet` library.

"""
from functools import partial
from typing import Callable

import jax.numpy as jnp
import pandas as pd
from jax import random
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon

from prophetverse.models import (univariate_gamma_model, univariate_model,
                                 univariate_negbinomial_model)
from prophetverse.sktime.base import (BaseBayesianForecaster,
                                      ExogenousEffectMixin)
from prophetverse.trend.flat import FlatTrend
from prophetverse.trend.piecewise import (PiecewiseLinearTrend,
                                          PiecewiseLogisticTrend, TrendModel)

__all__ = ["Prophet", "ProphetGamma", "ProphetNegBinomial"]


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
        offset_prior_scale (float): Scale parameter for the prior distribution of the offset. The offset is the constant term in the piecewise trend equation. Default is 0.1.
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
        exogenous_effects (List[AbstractEffect]): A list defining the exogenous effects to be used in the model.
        default_effect (AbstractEffect): The default effect to be used when no effect is specified for a variable.
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
        offset_prior_scale=0.1,
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
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):
        """
        Initializes the Prophet model.
        """

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend

        super().__init__(
            rng_key=rng_key,
            # ExogenousEffectMixin
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
            # BaseBayesianForecaster
            inference_method=inference_method,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            scale=scale,
        )

        self.model = univariate_model
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
        if self.offset_prior_scale <= 0:
            raise ValueError("offset_prior_scale must be greater than 0.")
        if self.trend not in ["linear", "logistic", "flat"] and not isinstance(self.trend, TrendModel):
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

        fh = y.index.get_level_values(-1).unique()

        self.trend_model_ = self._get_trend_model()
        self.trend_model_.initialize(y)

        trend_data = self.trend_model_.prepare_input_data(fh)

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
            
            "trend_model": self.trend_model_,
            "noise_scale" : self.noise_scale,
            "exogenous_effects": (
                self.exogenous_effect_dict if self._has_exogenous else None
            ),
        }

        inputs = {
            "y": y_array,
            "data": exogenous_data,
            "trend_data": trend_data,
            **self.fit_and_predict_data_,
        }

        return inputs

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

        trend_data = self.trend_model_.prepare_input_data(fh_as_index)

        if X is None:
            X = pd.DataFrame(index=fh_as_index)

        if self.feature_transformer is not None:
            X = self.feature_transformer.transform(X)

        exogenous_data = (
            self._get_exogenous_data_array(X.loc[fh_as_index]) if self._has_exogenous else None
        )

        return dict(
            y=None,
            data=exogenous_data,
            trend_data=trend_data,
            **self.fit_and_predict_data_,
        )

    def _get_trend_model(self):
        """
        Returns the trend model based on the specified trend parameter.

        Returns:
            TrendModel: The trend model based on the specified trend parameter.

        Raises:
            ValueError: If the trend parameter is not one of 'linear', 'logistic', 'flat' or a TrendModel instance.
        """

        ## Changepoints and trend
        if self.trend == "linear":
            return PiecewiseLinearTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
            )

        elif self.trend == "logistic":
            return PiecewiseLogisticTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
                capacity_prior=dist.TransformedDistribution(
                    dist.HalfNormal(self.capacity_prior_scale),
                    dist.transforms.AffineTransform(
                        loc=self.capacity_prior_loc, scale=1
                    ),
                ),
            )
        elif self.trend == "flat":
            return FlatTrend(
                changepoint_prior_scale=self.changepoint_prior_scale
            )

        elif isinstance(self.trend, TrendModel):
            return self.trend

        raise ValueError(
            "trend must be either 'linear', 'logistic' or a TrendModel instance."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):

        return [{
            "optimizer_steps": 1_000,
        }]


class ProphetGamma(Prophet):
    """A subclass of Prophet that models time series data with a gamma likelihood. Useful for positive only timeseries.

    Args:
        changepoint_interval (int): The number of points between potential changepoints. Default is 25.
        changepoint_range (float): Proportion of history in which trend changepoints will be estimated. Default is 0.8.
        changepoint_prior_scale (float): Parameter modulating the flexibility of the automatic changepoint selection. Default is 0.001.
        offset_prior_scale (float): Scale parameter for the prior distribution of the offset. Default is 0.1.
        feature_transformer (object): A transformer object to preprocess exogenous features. Default is None.
        capacity_prior_scale (float): Scale parameter for the capacity prior distribution. Default is 0.2.
        capacity_prior_loc (float): Location parameter for the capacity prior distribution. Default is 1.1.
        noise_scale (float): Scale parameter for the observation noise. Default is 0.05.
        trend (str): Type of trend component. Default is "logistic".
        mcmc_samples (int): Number of MCMC samples. Default is 2000.
        mcmc_warmup (int): Number of MCMC warmup steps. Default is 200.
        mcmc_chains (int): Number of MCMC chains. Default is 4.
        inference_method (str): Method for inference. Default is "map".
        optimizer_name (str): Name of the optimizer. Default is "Adam".
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer. Default is None.
        optimizer_steps (int): Number of optimizer steps. Default is 1000.
        exogenous_effects (list): List of exogenous effects. Default is None.
        default_effect (object): Default effect for exogenous features. Default is None.
        rng_key (jax.random.PRNGKey): Random number generator key. Default is None.
    """

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="logistic",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            feature_transformer=feature_transformer,
            capacity_prior_scale=capacity_prior_scale,
            capacity_prior_loc=capacity_prior_loc,
            noise_scale=noise_scale,
            trend=trend,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            exogenous_effects=exogenous_effects,
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
        )

        self.model = univariate_gamma_model


class ProphetNegBinomial(Prophet):
    """
    A subclass of Prophet that models time series data with a negative binomial likelihood. Useful for count data.

    Args:
        changepoint_interval (int): The number of points between potential changepoints. Default is 25.
        changepoint_range (float): Proportion of history in which trend changepoints will be estimated. Default is 0.8.
        changepoint_prior_scale (float): Parameter modulating the flexibility of the automatic changepoint selection. Default is 0.001.
        offset_prior_scale (float): Scale parameter for the prior distribution of the offset. Default is 0.1.
        feature_transformer (object): A transformer object to preprocess exogenous features. Default is None.
        capacity_prior_scale (float): Scale parameter for the capacity prior distribution. Default is 0.2.
        capacity_prior_loc (float): Location parameter for the capacity prior distribution. Default is 1.1.
        noise_scale (float): Scale parameter for the observation noise. Default is 0.05.
        trend (str): Type of trend component. Default is "logistic".
        mcmc_samples (int): Number of MCMC samples. Default is 2000.
        mcmc_warmup (int): Number of MCMC warmup steps. Default is 200.
        mcmc_chains (int): Number of MCMC chains. Default is 4.
        inference_method (str): Method for inference. Default is "map".
        optimizer_name (str): Name of the optimizer. Default is "Adam".
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer. Default is None.
        optimizer_steps (int): Number of optimizer steps. Default is 1000.
        exogenous_effects (list): List of exogenous effects. Default is None.
        default_effect (object): Default effect for exogenous features. Default is None.
        rng_key (jax.random.PRNGKey): Random number generator key. Default is None.
    """

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="logistic",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            feature_transformer=feature_transformer,
            capacity_prior_scale=capacity_prior_scale,
            capacity_prior_loc=capacity_prior_loc,
            noise_scale=noise_scale,
            trend=trend,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            exogenous_effects=exogenous_effects,
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
        )

        self.model = univariate_negbinomial_model

    def _scale_y(self, y : pd.DataFrame):
        """
        We skip the scaling of the data, as the negative binomial need integer values.
        
        However, we keep the timeseries scale to scale the changepoint coefficients, and parameters,
        for the latent mean.
        """
        return y

    def _inv_scale_y(self, y):
        """
        We skip the scaling of the data, as the negative binomial need integer values.

        However, we keep the timeseries scale to scale the changepoint coefficients, and parameters,
        for the latent mean.
        """
        return y

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

        fh = y.index.get_level_values(-1).unique()

        self.trend_model_ = self._get_trend_model()
        self.trend_model_.initialize(y/self._scale)

        trend_data = self.trend_model_.prepare_input_data(fh)

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
            "trend_model": self.trend_model_,
            "noise_scale": self.noise_scale,
            "exogenous_effects": (
                self.exogenous_effect_dict if self._has_exogenous else None
            ),
            "scale" : self._scale,
        }

        inputs = {
            "y": y_array,
            "data": exogenous_data,
            "trend_data": trend_data,
            **self.fit_and_predict_data_,
        }

        return inputs
