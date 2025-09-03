"""EmptyInputValidator implementation - validates that X is empty during fit."""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect

__all__ = ["EmptyInputValidator"]


class EmptyInputValidator(BaseEffect):
    """Effect that validates X is empty during fit and does nothing during predict.

    This effect is useful for ensuring that certain models or configurations
    are only used when no exogenous variables are provided.

    Raises an error if X passed to fit is not empty (i.e., has columns).
    Does nothing during prediction.
    """

    _tags = {
        "capability:panel": True,
        "capability:multivariate_input": True,
        "requires_X": True,  # We need X to validate it's empty
        "applies_to": "X",
        "filter_indexes_with_forecating_horizon_at_transform": False,
        "requires_fit_before_transform": False,
    }

    def __init__(self):
        """Initialize the EmptyInputValidator."""
        super().__init__()

    def _transform(self, X, fh):
        """Transform input data - return None since we don't use data for prediction."""
        # We already validated during fit, so we don't need the data for prediction
        return None

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Validate that X is empty (has no columns).

        Parameters
        ----------
        y : pd.DataFrame
            The target time series data.
        X : pd.DataFrame
            The exogenous variables DataFrame. Must be empty (no columns).
        scale : float, optional
            The scale factor, by default 1.0.

        Raises
        ------
        ValueError
            If X is not empty (has columns).
        """
        if X is not None and len(X.columns) > 0:
            raise ValueError(
                f"EmptyInputValidator requires X to be empty (no columns), "
                f"but X has {len(X.columns)} columns: {list(X.columns)}"
            )

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Return zeros with the appropriate shape.

        During prediction, this effect does nothing and returns zeros.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        jnp.ndarray
            An array of zeros with shape matching the forecast horizon.
            Shape is (T, 1) for univariate timeseries, or (N, T, 1) for
            multivariate/panel timeseries.
        """
        # Get shape from trend if available
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
