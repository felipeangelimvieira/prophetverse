"""Definition of Log Effect class."""

from typing import Optional

import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseEffect

__all__ = ["LogEffect"]


class LogEffect(BaseEffect):
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
        id: str = "",
        regex: Optional[str] = None,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        scale_prior: Optional[Distribution] = None,
        rate_prior: Optional[Distribution] = None,
        **kwargs,
    ):
        self.scale_prior = scale_prior or dist.Gamma(1, 1)
        self.rate_prior = rate_prior or dist.Gamma(1, 1)
        super().__init__(id=id, regex=regex, effect_mode=effect_mode, **kwargs)

    def _apply(  # type: ignore[override]
        self, trend: jnp.ndarray, data: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
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

        return effect
