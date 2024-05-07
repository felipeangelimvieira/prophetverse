import sys

#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict


import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from ..effects import AbstractEffect

def model(
    y,
    trend_model,
    trend_data={},
    data={},
    exogenous_effects: Dict[str, AbstractEffect]={},
    noise_scale=0.5,    
):
    """
    Defines the Numpyro model.

    Args:
        y (jnp.ndarray): Array of time series data.
        X (jnp.ndarray): Array of exogenous variables.
        t (jnp.ndarray): Array of time values.
    """
    trend = trend_model(**trend_data)


    numpyro.deterministic("trend_", trend)

    mean = trend
    # Exogenous effects
    if exogenous_effects is not None:

        for key, exog_effect in exogenous_effects.items():

            exog_data = data[key]
            effect = exog_effect(trend=trend, data=exog_data)
            numpyro.deterministic(key, effect)
            mean += effect

    noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(noise_scale))

    with numpyro.plate("data", len(mean), dim=-2) as time_plate:
        numpyro.sample(
            "obs",
            dist.Normal(mean.reshape((-1, 1)), noise_scale),
            obs=y,
        )
