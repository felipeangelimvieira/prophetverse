"""Definition of Linear Effect class."""

from typing import Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    EFFECT_APPLICATION_TYPE,
    BaseAdditiveOrMultiplicativeEffect,
)
from prophetverse.utils.algebric_operations import matrix_multiplication

__all__ = ["LinearEffect"]


class LinearEffect(BaseAdditiveOrMultiplicativeEffect):
    """Represents a linear effect in a hierarchical prophet model.

    Parameters
    ----------
    prior : Distribution, optional
        A numpyro distribution to use as prior. Defaults to dist.Normal(0, 1)
    effect_mode : effects_application, optional
        Either "multiplicative" or "additive" by default "multiplicative".
    """

    _tags = {
        "supports_multivariate": True,
    }

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        prior: Optional[Distribution] = None,
    ):
        self.prior = prior or dist.Normal(0, 0.1)

        super().__init__(effect_mode=effect_mode)

    def _predict(self, trend: jnp.ndarray, **kwargs) -> jnp.ndarray:
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
        data = kwargs.pop("data")

        n_features = data.shape[-1]

        with numpyro.plate("features_plate", n_features, dim=-1):
            coefficients = numpyro.sample("coefs", self.prior)

        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)

        return matrix_multiplication(data, coefficients)
