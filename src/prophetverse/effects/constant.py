from prophetverse.effects.base import BaseEffect
import pandas as pd
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


class Constant(BaseEffect):
    """
    Constant effect
    """

    _tags = {
        "requires_X": False,
    }

    def __init__(self, prior: float = None) -> None:
        self.prior = prior
        super().__init__()

        self._prior = prior
        if self._prior is None:
            self._prior = dist.Normal(0, 1)

    def _transform(self, X, fh):
        return jnp.ones((len(fh), 1))

    def _predict(  # type: ignore[override]
        self, data: jnp.ndarray, predicted_effects: dict, *args, **kwargs
    ) -> jnp.ndarray:
        """Apply the trend.

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

        coefficient = numpyro.sample("constant_coefficient", self._prior)

        return coefficient * jnp.ones_like(data)
