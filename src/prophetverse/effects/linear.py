"""Definition of Linear Effect class."""

from typing import Any, Dict, Optional

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
        "capability:panel": True,
        "capability:multivariate_input": True,
    }

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        prior: Optional[Distribution] = None,
        broadcast=False,
    ):
        self.prior = prior
        self.broadcast = broadcast
        self._prior = self.prior if prior is not None else dist.Normal(0, 0.1)

        super().__init__(effect_mode=effect_mode)

        if self.broadcast:
            self.set_tags(**{"capability:multivariate_input": False})

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
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
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        n_features = data.shape[-1]

        with numpyro.plate("features_plate", n_features, dim=-1):
            coefficients = numpyro.sample("coefs", self._prior)

        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)

        return data @ coefficients
