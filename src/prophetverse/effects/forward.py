"""Constant effect module."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import BaseEffect


class Forward(BaseEffect):
    """Forward effect.

    Forwards a previously fitted effect.

    Parameters
    ----------
    prior : Distribution, optional
        The prior distribution for the constant coefficient, by default None
        which corresponds to a standard normal distribution.
    """

    _tags = {
        "requires_X": False,
    }

    def __init__(self, effect_name: str) -> None:
        self.effect_name = effect_name
        super().__init__()

    def _predict(  # type: ignore[override]
        self, data: jnp.ndarray, predicted_effects: dict, *args, **kwargs
    ) -> jnp.ndarray:
        """Forwards the effect

        Parameters
        ----------
        constant_vector : jnp.ndarray
            A constant vector with the size of the series time indexes

        Returns
        -------
        jnp.ndarray
            The forecasted trend
        """
        # Alias for clarity

        return predicted_effects[self.effect_name]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"effect_name": "trend"}]
