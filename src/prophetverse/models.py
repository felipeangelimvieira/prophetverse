from typing import Dict

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from prophetverse.distributions import GammaReparametrized
from prophetverse.effects import AbstractEffect
from prophetverse.trend.base import TrendModel


def multivariate_model(
    y,
    trend_model : TrendModel,
    trend_data: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray] = None,
    exogenous_effects: Dict[str, AbstractEffect] = None,
    noise_scale=0.05,
    correlation_matrix_concentration=1.0,
    is_single_series=False,
):
    """
    Defines the Numpyro multivariate model.
    
    The multivariate model is infers a Prophet-like model for each time series and use
    a multivariate normal likelihood as the observation model.
    
    Args:
        y (jnp.ndarray): Array of time series data.
        trend_model (TrendModel): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
        correlation_matrix_concentration (float): Concentration parameter for the LKJ distribution.
        
    """
    
    trend = trend_model(**trend_data)

    numpyro.deterministic("trend", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for key, exog_effect in exogenous_effects.items():

            exog_data = data[key]
            effect = exog_effect(trend=trend, data=exog_data)
            numpyro.deterministic(key, effect)
            mean += effect

    std_observation = numpyro.sample(
        "std_observation", dist.HalfNormal(jnp.array([noise_scale] * mean.shape[0]))
    )

    if y is not None:
        y = y.squeeze(-1).T

    if is_single_series:

        with numpyro.plate("time", mean.shape[-1], dim=-2):
            numpyro.sample(
                "obs", dist.Normal(mean.squeeze(-1).T, std_observation), obs=y
            )

    else:
        correlation_matrix = numpyro.sample(
            "corr_matrix",
            dist.LKJCholesky(
                mean.shape[0],
                concentration=correlation_matrix_concentration,
            ),
        )

        cov_mat = (
            jnp.diag(std_observation) @ correlation_matrix @ jnp.diag(std_observation)
        )

        cov_mat = jnp.tile(jnp.expand_dims(cov_mat, axis=0), (mean.shape[1], 1, 1))

        with numpyro.plate("time", mean.shape[-1], dim=-2):
            numpyro.sample(
                "obs",
                dist.MultivariateNormal(mean.squeeze(-1).T, scale_tril=cov_mat),
                obs=y,
            )


def univariate_model(
    y,
    trend_model: TrendModel,
    trend_data: Dict[str, jnp.ndarray], 
    data: Dict[str, jnp.ndarray] = None,
    exogenous_effects: Dict[str, AbstractEffect] = None,
    noise_scale=0.5,
):
    """
    Defines the Prophet-like model for univariate timeseries.

    Args:
        y (jnp.ndarray): Array of time series data.
        trend_model (TrendModel): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
    """

    mean = _compute_mean_univariate(
        trend_model=trend_model,
        trend_data=trend_data,
        data=data,
        exogenous_effects=exogenous_effects,
    )

    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2) as time_plate:
        numpyro.sample(
            "obs",
            dist.Normal(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )


def univariate_gamma_model(
    y,
    trend_model: TrendModel,
    trend_data: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray] = None,
    exogenous_effects: Dict[str, AbstractEffect] = None,
    noise_scale=0.5,
):
    """
    Defines the Prophet-like model for univariate timeseries.

    Args:
        y (jnp.ndarray): Array of time series data.
        trend_model (TrendModel): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
    """

    mean = _compute_mean_univariate(
        trend_model=trend_model,
        trend_data=trend_data,
        data=data,
        exogenous_effects=exogenous_effects,
    )

    mean = _to_positive(mean, 1e-5)

    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2) as time_plate:
        numpyro.sample(
            "obs",
            GammaReparametrized(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )


def univariate_negbinomial_model(
    y,
    trend_model: TrendModel,
    trend_data: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray] = None,
    exogenous_effects: Dict[str, AbstractEffect] = None,
    noise_scale=0.5,
    scale=1,
):
    """
    Defines the Prophet-like model for univariate timeseries.

    Args:
        y (jnp.ndarray): Array of time series data.
        trend_model (TrendModel): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
    """

    mean = _compute_mean_univariate(
        trend_model=trend_model,
        trend_data=trend_data,
        data=data,
        exogenous_effects=exogenous_effects,
    )
    
    
    
    mean = _to_positive(mean, 1e-5)
    
    mean = mean*scale
    
    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2) as time_plate:
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean.reshape((-1, 1)), noise_scale * scale),
            obs=y,
        )


def _to_positive(x, threshold):
    return jnp.where(x < threshold, jnp.exp(x - threshold) * threshold, x)


def _compute_mean_univariate(
    trend_model: TrendModel,
    trend_data: Dict[str, jnp.ndarray], 
    data: Dict[str, jnp.ndarray] = None,
    exogenous_effects: Dict[str, AbstractEffect] = None,
    ):
    trend = trend_model(**trend_data)

    numpyro.deterministic("trend", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for key, exog_effect in exogenous_effects.items():

            exog_data = data[key]
            effect = exog_effect(trend=trend, data=exog_data)
            numpyro.deterministic(key, effect)
            mean += effect
    return mean
