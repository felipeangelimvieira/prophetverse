"""BypassEffect implementation - ignores inputs and returns zeros."""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect

__all__ = ["IgnoreInput"]


class IgnoreInput(BaseEffect):
    """Effect that ignores all inputs and returns zeros during prediction.

    This effect can be used as a placeholder or to disable specific effects
    without removing them from the model configuration.

    The effect ignores all input data and always returns zeros with the
    appropriate shape for the forecast horizon.

    Parameters
    ----------
    raise_error : bool, optional
        If True, validates that X is empty (has no columns) during fit.
        If False, ignores X completely. Default is False.
    """

    _tags = {
        "requires_X": True,  # Default value, will be overridden in __init__
        "applies_to": "X",
    }

    def __init__(self, raise_error: bool = False):
        """Initialize the BypassEffect.

        Parameters
        ----------
        validate_empty_input : bool, optional
            If True, validates that X is empty (has no columns) during fit.
            If False, ignores X completely. Default is False.
        """
        self.raise_error = raise_error
        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Fit the effect. If validation is enabled, check that X is empty.

        Parameters
        ----------
        y : pd.DataFrame
            The target time series data.
        X : pd.DataFrame
            The exogenous variables DataFrame.
        scale : float, optional
            The scale factor, by default 1.0.

        Raises
        ------
        ValueError
            If validate_empty_input is True and X is not empty (has columns).
        """
        if self.raise_error and X is not None and len(X.columns) > 0:
            raise ValueError(
                f"BypassEffect with raise_error=True requires X to be empty "
                f"(no columns), but X has {len(X.columns)} columns: {list(X.columns)}"
            )

    def _transform(self, X, fh):
        if X is None:
            return None
        return super()._transform(X, fh)

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Return zero.

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
        if data is None:
            return 0
        return jnp.zeros((data.shape[0], 1))
