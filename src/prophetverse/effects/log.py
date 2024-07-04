"""Definition of Log Effect class."""

from typing import Optional

import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import AbstractEffect
from prophetverse.effects.effect_apply import EFFECT_APPLICATION_TYPE

__all__ = ["LogEffect"]


class LogEffect(AbstractEffect):
    """Represents a log effect as effect = scale * log(rate * data + 1).

    Parameters
    ----------
    scale_prior : Optional[Distribution], optional
        The prior distribution for the scale parameter., by default Gamma
    rate_prior : Optional[Distribution], optional
        The prior distribution for the rate parameter., by default Gamma
    effect_mode : effects_application, optional
        Either "additive" or "multiplicative", by default "multiplicative"
    """

    def __init__(
        self,
        scale_prior: Optional[Distribution] = None,
        rate_prior: Optional[Distribution] = None,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        **kwargs,
    ):
        self.scale_prior = scale_prior or dist.Gamma(1, 1)
        self.rate_prior = rate_prior or dist.Gamma(1, 1)
        self.effect_mode = effect_mode
        super().__init__(**kwargs)

    def compute_effect(self, trend: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Compute the effect using the log transformation.

        Parameters
        ----------
        trend : jnp.ndarray
            The trend component.
        data : jnp.ndarray
            The input data.

        Returns
        -------
        jnp.ndarray
            The computed effect based on the given trend and data.
        """
        scale = self.sample("log_scale", self.scale_prior)
        rate = self.sample("log_rate", self.rate_prior)

        if jnp.any(rate * data + 1 <= 0):
            raise ValueError("Can't take log of negative values or zero.")

        effect = scale * jnp.log(rate * data + 1)

        if self.effect_mode == "additive":
            return effect
        return trend * effect
