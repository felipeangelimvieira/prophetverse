"""Defines the numpyro models used in the Prophet-like framework."""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from prophetverse.distributions import GammaReparametrized
from prophetverse.effects.base import BaseEffect


def multivariate_model(
    y,
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
    noise_scale=0.05,
    correlation_matrix_concentration=1.0,
    is_single_series=False,
    **kwargs,
):
    """
    Define the Numpyro multivariate model.

    The multivariate model is infers a Prophet-like model for each time series and use
    a multivariate normal likelihood as the observation model.

    Parameters
    ----------
        y (jnp.ndarray): Array of time series data.
        trend_model (BaseEffect): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
        correlation_matrix_concentration (float): Concentration parameter for the LKJ
        distribution.
    """
    mean = _compute_mean_univariate(
        trend_model=trend_model,
        trend_data=trend_data,
        data=data,
        exogenous_effects=exogenous_effects,
    )

    if y is not None:
        y = y.squeeze(-1).T

    mean = numpyro.deterministic("mean", mean)
    if is_single_series:

        mean = mean.reshape((-1, 1))
        std_observation = numpyro.sample(
            "std_observation", dist.HalfNormal(jnp.array(noise_scale))
        )

        with numpyro.plate("time", mean.shape[-1], dim=-2):
            numpyro.sample("obs", dist.Normal(mean, std_observation), obs=y)

    else:

        std_observation = numpyro.sample(
            "std_observation", dist.HalfNormal(jnp.array([noise_scale] * mean.shape[0]))
        )

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
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
    noise_scale=0.5,
    **kwargs,
):
    """
    Define the Prophet-like model for univariate timeseries.

    Parameters
    ----------
        y (jnp.ndarray): Array of time series data.
        trend_model (BaseEffect): Trend model.
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
    mean = numpyro.deterministic("mean", mean)
    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2):
        numpyro.sample(
            "obs",
            dist.Normal(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )


def univariate_gamma_model(
    y,
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
    noise_scale=0.5,
    **kwargs,
):
    """
    Define the Prophet-like model for univariate timeseries.

    Parameters
    ----------
        y (jnp.ndarray): Array of time series data.
        trend_model (BaseEffect): Trend model.
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
    mean = numpyro.deterministic("mean", mean)
    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2):
        numpyro.sample(
            "obs",
            GammaReparametrized(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )


def univariate_negbinomial_model(
    y,
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
    noise_scale=0.5,
    scale=1,
    **kwargs,
):
    """
    Define the Prophet-like model for univariate timeseries.

    Parameters
    ----------
        y (jnp.ndarray): Array of time series data.
        trend_model (BaseEffect): Trend model.
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

    mean = mean * scale
    mean = numpyro.deterministic("mean", mean)
    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2):
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean.reshape((-1, 1)), noise_scale * scale),
            obs=y,
        )


def _to_positive(
    x: jnp.ndarray, smooth_threshold: float, threshold: float = 1e-10
) -> jnp.ndarray:
    """Force the values of x to be positive.

    Applies a smooth threshold to the values of x to force positive-only outputs.
    Further clips the values of x to avoid zeros due to numerical precision.

    Parameters
    ----------
    x : jnp.ndarray
        The array to be transformed.
    smooth_threshold : float
        The threshold value for the exponential function.
    threshold : float, optional
        The threshold value for clipping, by default 1e-10.

    Returns
    -------
    jnp.ndarray
        The transformed array.
    """
    return jnp.clip(
        jnp.where(
            x < smooth_threshold, jnp.exp(x - smooth_threshold) * smooth_threshold, x
        ),
        threshold,
        None,
    )


def _compute_mean_univariate(
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
):

    predicted_effects: Dict[str, jnp.ndarray] = {}

    with numpyro.handlers.scope(prefix="trend"):
        trend = trend_model(data=trend_data, predicted_effects=predicted_effects)

    predicted_effects["trend"] = trend

    numpyro.deterministic("trend", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for exog_effect_name, exog_effect in exogenous_effects.items():
            transformed_data = data[exog_effect_name]  # type: ignore[index]
            with numpyro.handlers.scope(prefix=exog_effect_name):
                effect = exog_effect(transformed_data, predicted_effects)
            effect = numpyro.deterministic(exog_effect_name, effect)
            mean += effect
            predicted_effects[exog_effect_name] = effect
    return mean
