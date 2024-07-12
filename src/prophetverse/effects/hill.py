"""Definition of Hill Effect class."""

from typing import Optional

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

    def _predict(self, trend: jnp.ndarray, **kwargs) -> jnp.ndarray:
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
        data: jnp.ndarray = kwargs.pop("data")

        half_max = numpyro.sample("half_max", self.half_max_prior)
        slope = numpyro.sample("slope", self.slope_prior)
        max_effect = numpyro.sample("max_effect", self.max_effect_prior)

        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)
        return effect
