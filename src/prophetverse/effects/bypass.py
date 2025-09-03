"""BypassEffect implementation - ignores inputs and returns zeros."""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect

__all__ = ["BypassEffect"]


class BypassEffect(BaseEffect):
    """Effect that ignores all inputs and returns zeros during prediction.

    This effect can be used as a placeholder or to disable specific effects
    without removing them from the model configuration.

    The effect ignores all input data and always returns zeros with the
    appropriate shape for the forecast horizon.

    Parameters
    ----------
    validate_empty_input : bool, optional
        If True, validates that X is empty (has no columns) during fit.
        If False, ignores X completely. Default is False.
    """

    _tags = {
        "capability:panel": True,
        "capability:multivariate_input": True,
        "requires_X": False,  # Default value, will be overridden in __init__
        "applies_to": "X",
        "filter_indexes_with_forecating_horizon_at_transform": False,
        "requires_fit_before_transform": False,
    }

    def __init__(self, validate_empty_input: bool = False):
        """Initialize the BypassEffect.

        Parameters
        ----------
        validate_empty_input : bool, optional
            If True, validates that X is empty (has no columns) during fit.
            If False, ignores X completely. Default is False.
        """
        self.validate_empty_input = validate_empty_input
        super().__init__()

        # Set tags based on validation mode
        self.set_tags(requires_X=self.validate_empty_input)

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
        if self.validate_empty_input and X is not None and len(X.columns) > 0:
            raise ValueError(
                f"BypassEffect with validate_empty_input=True requires X to be empty "
                f"(no columns), but X has {len(X.columns)} columns: {list(X.columns)}"
            )

    def _transform(self, X, fh):
        """Transform input data - handle None and empty DataFrames properly."""
        if X is None:
            return None
        if isinstance(X, pd.DataFrame) and len(X.columns) == 0:
            return None
        # Use default behavior for non-empty X
        from prophetverse.utils.frame_to_array import series_to_tensor_or_array

        return series_to_tensor_or_array(X)

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
        return jnp.array(0.0)
