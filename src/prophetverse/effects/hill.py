"""Definition of Hill Effect class."""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    EFFECT_APPLICATION_TYPE,
    BaseAdditiveOrMultiplicativeEffect,
)
from prophetverse.utils.algebric_operations import _exponent_safe

__all__ = ["HillEffect"]


class HillEffect(BaseAdditiveOrMultiplicativeEffect):
    """Represents a Hill effect in a time series model.

    Parameters
    ----------
    half_max_prior : Distribution, optional
        Prior distribution for the half-maximum parameter
    slope_prior : Distribution, optional
        Prior distribution for the slope parameter
    max_effect_prior : Distribution, optional
        Prior distribution for the maximum effect parameter
    effect_mode : effects_application, optional
        Mode of the effect (either "additive" or "multiplicative")
    """

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        half_max_prior: Optional[Distribution] = None,
        slope_prior: Optional[Distribution] = None,
        max_effect_prior: Optional[Distribution] = None,
    ):
        self.half_max_prior = half_max_prior or dist.Gamma(1, 1)
        self.slope_prior = slope_prior or dist.HalfNormal(10)
        self.max_effect_prior = max_effect_prior or dist.Gamma(1, 1)

        super().__init__(effect_mode=effect_mode)

    def _sample_params(
        self, data, predicted_effects: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample the parameters of the effect.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.
        """
        return {
            "half_max": numpyro.sample("half_max", self.half_max_prior),
            "slope": numpyro.sample("slope", self.slope_prior),
            "max_effect": numpyro.sample("max_effect", self.max_effect_prior),
        }

    def _predict(
        self,
        data: Dict[str, jnp.ndarray],
        predicted_effects: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries.
        """
        half_max = params["half_max"]
        slope = params["slope"]
        max_effect = params["max_effect"]

        data = jnp.clip(data, 1e-9, None)
        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)
        return effect
