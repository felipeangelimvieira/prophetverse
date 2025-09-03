"""BypassEffect implementation - ignores inputs and returns zeros."""

from typing import Any, Dict, Optional

import jax.numpy as jnp

from prophetverse.effects.base import BaseEffect

__all__ = ["BypassEffect"]


class BypassEffect(BaseEffect):
    """Effect that ignores all inputs and returns zeros during prediction.

    This effect can be used as a placeholder or to disable specific effects
    without removing them from the model configuration.

    The effect ignores all input data and always returns zeros with the
    appropriate shape for the forecast horizon.
    """

    _tags = {
        "capability:panel": True,
        "capability:multivariate_input": True,
        "requires_X": False,  # Don't require X since we ignore it anyway
        "applies_to": "X",
        "filter_indexes_with_forecating_horizon_at_transform": False,
        "requires_fit_before_transform": False,
    }

    def __init__(self):
        """Initialize the BypassEffect."""
        super().__init__()

    def _transform(self, X, fh):
        """Transform input data - return None since we ignore inputs anyway."""
        # We don't need the actual data since we ignore it in _predict
        return None

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Return zeros with the appropriate shape.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method (ignored).

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        jnp.ndarray
            An array of zeros with shape matching the forecast horizon.
            Shape is (T, 1) for univariate timeseries, or (N, T, 1) for
            multivariate/panel timeseries.
        """
        # Get shape from trend if available, otherwise use data shape
        if "trend" in predicted_effects:
            return jnp.zeros_like(predicted_effects["trend"])

        # Fallback: try to infer shape from data
        if data is not None:
            if isinstance(data, jnp.ndarray):
                if data.ndim == 3:  # Panel data (N, T, features)
                    return jnp.zeros((data.shape[0], data.shape[1], 1))
                elif data.ndim == 2:  # Single series (T, features)
                    return jnp.zeros((data.shape[0], 1))

        # Final fallback: return scalar zero
        return jnp.zeros((1, 1))
