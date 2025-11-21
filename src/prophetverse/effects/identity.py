"""Store input to predicted_effect"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect

__all__ = ["Identity"]


class Identity(BaseEffect):
    """
    Return the input as the predicted effect.

    This effect simply returns the input data during prediction without any
    modification.
    """

    _tags = {
        "requires_X": True,  # Default value, will be overridden in __init__
        "applies_to": "X",
    }

    def __init__(self):

        super().__init__()

    def _transform(self, X, fh):
        if X is None:
            raise ValueError("Input X cannot be None in _transform method.")
        return super()._transform(X, fh)

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Return the data

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method (ignored).

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects (ignored).

        Returns
        -------
        jnp.ndarray
            Zero.
        """
        return data
