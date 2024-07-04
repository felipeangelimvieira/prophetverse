"""Definition of Hill Effect class."""

from typing import Optional

import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import AbstractEffect
from prophetverse.effects.effect_apply import EFFECT_APPLICATION_TYPE
from prophetverse.utils.algebric_operations import _exponent_safe

__all__ = ["HillEffect"]


class HillEffect(AbstractEffect):
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
        half_max_prior: Optional[Distribution] = None,
        slope_prior: Optional[Distribution] = None,
        max_effect_prior: Optional[Distribution] = None,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        **kwargs,
    ):
        self.half_max_prior = half_max_prior or dist.Gamma(1, 1)
        self.slope_prior = slope_prior or dist.HalfNormal(10)
        self.max_effect_prior = max_effect_prior or dist.Gamma(1, 1)
        self.effect_mode = effect_mode
        super().__init__(**kwargs)

    def compute_effect(self, trend: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Compute the effect using the log transformation.

        Parameters
        ----------
        trend : jnp.ndarray
            The trend component of the hierarchical prophet model.
        data : jnp.ndarray
            The data used to compute the effect.

        Returns
        -------
        jnp.ndarray
            The computed effect based on the given trend and data.
        """
        half_max = self.sample("half_max", self.half_max_prior)
        slope = self.sample("slope", self.slope_prior)
        max_effect = self.sample("max_effect", self.max_effect_prior)

        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)

        if self.effect_mode == "additive":
            return effect
        return trend * effect
