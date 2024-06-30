"""Definition of Linear Effect class."""

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import AbstractEffect
from prophetverse.effects.effect_apply import (
    additive_effect,
    effects_application,
    multiplicative_effect,
)


class LinearEffect(AbstractEffect):
    """Represents a linear effect in a hierarchical prophet model."""

    def __init__(
        self,
        prior: None | Distribution = None,
        effect_mode: effects_application = "multiplicative",
        **kwargs,
    ):
        """Initialize linear effect priors and application.

        Parameters
        ----------
        prior : None | Distribution, optional
            A numpyro distribution to use as prior. Defaults to dist.Normal(0, 1)
        effect_mode : effects_application, optional
            Either "multiplicative" or "additive" by default "multiplicative".
        """
        self.prior = prior or dist.Normal(0, 0.1)
        self.effect_mode = effect_mode

        super().__init__(**kwargs)

    def compute_effect(self, trend: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Compute the Linear effect.

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
        n_features = data.shape[-1]

        with numpyro.plate(f"{self.id}_plate", n_features, dim=-1):
            coefficients = self.sample("coefs", self.prior)

        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)
        if self.effect_mode == "multiplicative":
            return multiplicative_effect(trend, data, coefficients)
        return additive_effect(data, coefficients)
