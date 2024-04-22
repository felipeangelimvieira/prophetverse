import sys

#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict


import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from .effects import EffectFunc, SampleParamsFunc

class ExogenousEffects(TypedDict):
    data: jnp.ndarray
    transformation_func: EffectFunc
    sample_params_func: SampleParamsFunc


def model(
    t,
    y,
    changepoint_matrix,
    init_trend_params,
    trend_mode,
    exogenous_effects: Dict[str, ExogenousEffects],
    y_scale: float,
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

    trend = (changepoint_matrix) @ changepoint_coefficients.reshape((-1, 1)) + offset
    trend = trend * y_scale
    
    numpyro.deterministic("trend_", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for key, exog_effects in exogenous_effects.items():
            sample_params_func = exog_effects["sample_params_func"]
            transform_func = exog_effects["transformation_func"]
            exog_data = exog_effects["data"]
            # TODO: add y scale as argument
            coefficients = sample_params_func()
            effect = transform_func(
                trend=trend, data=exog_data, coefficients=coefficients
            )
            numpyro.deterministic(key, effect)
            mean += effect

    noise_scale = params["std_observation"] * y_scale

    with numpyro.plate("data", len(mean), dim=-2) as time_plate:
        numpyro.sample(
            "obs",
            dist.Normal(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )
