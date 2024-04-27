import sys

#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict


import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from hierarchical_prophet.effects import AbstractEffect
from hierarchical_prophet.hierarchical_prophet._distribution import NormalReconciled


def model(
    t,
    y,
    changepoint_matrix,
    init_trend_params,
    trend_mode,
    data={},
    exogenous_effects: Dict[str, AbstractEffect] = {},
    noise_scale=0.05,
    y_scale: float = 1,
):
    """
    Defines the Numpyro model.

    Args:
        y (jnp.ndarray): Array of time series data.
        X (jnp.ndarray): Array of exogenous variables.
        t (jnp.ndarray): Array of time values.
    """
    params = init_trend_params()

    # Trend
    changepoint_coefficients = params["changepoint_coefficients"]
    offset = params["offset"]
    capacity = params.get("capacity", None)

    trend = (changepoint_matrix) @ changepoint_coefficients.reshape(
        (1, -1, 1)
    ) + offset.reshape((-1, 1, 1))
    if trend_mode == "logistic":
        trend = capacity.reshape((1, -1, 1)) / (1 + jnp.exp(-trend))

    trend = trend * y_scale.reshape((-1, 1, 1))

    numpyro.deterministic("trend_", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for key, exog_effect in exogenous_effects.items():

            exog_data = data[key]
            effect = exog_effect(trend=trend, data=exog_data)
            numpyro.deterministic(key, effect)
            mean += effect

    std_observation = numpyro.sample(
        "std_observation", dist.HalfNormal(jnp.array([0.1] * mean.shape[0]))
    )

    correlation_matrix = numpyro.sample(
        "corr_matrix",
        dist.LKJCholesky(
            mean.shape[0],
            concentration=1.0,
        ),
    )

    noise_scale = std_observation * y_scale

    cov_mat = jnp.diag(noise_scale) @ correlation_matrix @ jnp.diag(noise_scale)

    cov_mat = jnp.tile(jnp.expand_dims(cov_mat, axis=0), (mean.shape[1], 1, 1))

    numpyro.sample(
        "obs",
        dist.MultivariateNormal(mean.squeeze(-1).T, scale_tril=cov_mat),
        obs=y,
    )
