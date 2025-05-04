import numpyro
from prophetverse.effects.base import BaseEffect
from typing import Any, Dict, Optional
import jax.numpy as jnp
from prophetverse.utils.numpyro import CacheMessenger


def wrap_with_cache_messenger(model):

    def wrapped(*args, **kwargs):
        with CacheMessenger():
            return model(*args, **kwargs)

    return wrapped


def model(
    y,
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    target_model: BaseEffect,
    target_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
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

    with CacheMessenger():
        predicted_effects: Dict[str, jnp.ndarray] = {}

        with numpyro.handlers.scope(prefix="trend"):
            trend = trend_model(data=trend_data, predicted_effects=predicted_effects)

        predicted_effects["trend"] = trend

        numpyro.deterministic("trend", trend)
        # Exogenous effects
        if exogenous_effects is not None:

            for exog_effect_name, exog_effect in exogenous_effects.items():
                transformed_data = data[exog_effect_name]  # type: ignore[index]
                with numpyro.handlers.scope(prefix=exog_effect_name):
                    effect = exog_effect(transformed_data, predicted_effects)
                effect = numpyro.deterministic(exog_effect_name, effect)

                predicted_effects[exog_effect_name] = effect

        target_model.predict(target_data, predicted_effects)
