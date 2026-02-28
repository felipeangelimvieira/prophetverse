"""Definition of Chained Effects class."""

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import pandas as pd
from skbase.base import BaseMetaEstimatorMixin
import numpyro
from prophetverse.effects.base import BaseEffect
from prophetverse.effects.adstock import BaseAdstockEffect

__all__ = ["ChainedEffects"]


class ChainedEffects(BaseMetaEstimatorMixin, BaseEffect):
    """
    Chains multiple effects sequentially, applying them one after the other.

    Parameters
    ----------
    steps : List[BaseEffect]
        A list of effects to be applied sequentially.

    Notes
    -----
    For budget optimization, ChainedEffects supports adstock in any position.
    When adstock is in a non-first position (e.g., Hill → Adstock), the chain
    automatically includes full historical data so that:
    1. The first effect (Hill) transforms all historical data
    2. Adstock receives the full saturated history for correct carryover
    3. Only horizon rows are selected for the final output
    """

    _tags = {
        "hierarchical_prophet_compliant": True,
        "requires_X": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "named_object_parameters": "named_steps",
        "capability:budget_optimization": True,
    }

    def __init__(self, steps: List[Tuple[str, BaseEffect]]):
        self.steps = steps
        super().__init__()

        self.named_steps = []
        for i, val in enumerate(self.steps):
            if isinstance(val, tuple):
                self.named_steps.append(val)
            elif isinstance(val, BaseEffect):
                self.named_steps.append((str(i), val))
            else:
                raise ValueError(
                    f"Invalid type {type(val)} for step {i}. Must be a tuple or BaseEffect."
                )

        # Propagate tags from inner effects
        all_panel = all(
            effect.get_tag("capability:panel", False)
            for _, effect in self.named_steps
        )
        any_hyperpriors = any(
            effect.get_tag("feature:panel_hyperpriors", False)
            for _, effect in self.named_steps
        )

        self.set_tags(
            **{
                "requires_X": steps[0][1].get_tag("requires_X", False),
                "capability:panel": all_panel,
                "feature:panel_hyperpriors": any_hyperpriors,
            }
        )

    def _fit(self, y: Any, X: Any, scale: float = 1.0):
        """
        Fit all chained effects sequentially.

        Parameters
        ----------
        y : Any
            Target data (e.g., time series values).
        X : Any
            Exogenous variables.
        scale : float, optional
            Scale of the timeseries.
        """
        # Store the fitted X for historical data access during transform
        self._fitted_X = X.copy() if hasattr(X, "copy") else X

        self.named_steps_ = []
        for name, effect in self.named_steps:
            effect = effect.clone()
            effect.fit(y, X, scale)
            self.named_steps_.append((name, effect))

    def _has_adstock_after_first(self) -> bool:
        """Check if any effect after the first is an adstock effect."""
        from prophetverse.effects.adstock import BaseAdstockEffect

        for i, (name, effect) in enumerate(self.named_steps_):
            if i > 0 and isinstance(effect, BaseAdstockEffect):
                return True
        return False

    def _transform(self, X: Any, fh: Any) -> Any:
        """
        Transform input data for the chain.

        When adstock effects are in non-first positions, we include full
        historical data so the first effect can transform all of it, enabling
        correct adstock carryover computation.

        Parameters
        ----------
        X : Any
            Input data (e.g., exogenous variables).
        fh : Any
            Forecasting horizon.

        Returns
        -------
        Any
            Transformed data - either simple transform output or dict with
            full historical transform if adstock effects need it.
        """
        # Check if we need to include historical data for later adstock effects
        if not self._has_adstock_after_first():
            # Simple case: just transform with first effect
            return self.named_steps_[0][1].transform(X, fh)

        # Build full historical X by combining fitted X with current X
        historical_X = X
        if hasattr(self, "_fitted_X") and self._fitted_X is not None:
            dates_to_add = self._fitted_X.index.difference(X.index)
            if len(dates_to_add) > 0:
                historical_X = pd.concat(
                    [self._fitted_X.loc[dates_to_add, X.columns], X], axis=0
                )
                historical_X = historical_X.sort_index()

        # Get indices to select horizon rows from full history
        horizon_indices = historical_X.index.get_indexer(X.index)
        horizon_indices = jnp.array(horizon_indices, dtype=jnp.int32)

        # Transform the FULL historical data through the first effect
        # We pass the full historical index as fh so the first effect doesn't filter
        full_fh = historical_X.index.get_level_values(-1)
        first_transform_full = self.named_steps_[0][1].transform(historical_X, full_fh)

        return {
            "first_transform": first_transform_full,
            "horizon_indices": horizon_indices,
        }

    def _update_data(self, data: Any, arr: jnp.ndarray) -> Any:
        """
        Update the chain's transform data with new input values.

        Parameters
        ----------
        data : Any
            The data structure from transform.
        arr : jnp.ndarray
            The new input array values.

        Returns
        -------
        Any
            Updated data structure with new values.
        """
        # Check if named_steps_ exists (set during fit)
        if not hasattr(self, "named_steps_"):
            # Fallback to base implementation if not fitted
            return super()._update_data(data, arr)

        if not isinstance(data, dict) or "first_transform" not in data:
            # Simple case: delegate to first effect
            return self.named_steps_[0][1]._update_data(data, arr)

        # Complex case: update the first_transform at horizon indices
        first_transform = data["first_transform"]
        horizon_indices = data["horizon_indices"]

        # Update the first transform using the first effect's _update_data
        # The arr contains new values for horizon positions
        # We need to update the full historical transform at those positions
        updated_first = self.named_steps_[0][1]._update_data(first_transform, arr)

        return {
            "first_transform": updated_first,
            "horizon_indices": horizon_indices,
        }

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Apply all chained effects sequentially.

        Parameters
        ----------
        data : Any
            Data from transform.
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        jnp.ndarray
            The transformed data after applying all effects.
        """

        # Handle dict format with full historical transform
        if isinstance(data, dict) and "first_transform" in data:
            first_transform = data["first_transform"]
            horizon_indices = data["horizon_indices"]

            output = None
            for i, (name, effect) in enumerate(self.named_steps_):
                with numpyro.handlers.scope(prefix=name):
                    if i == 0:
                        # First effect predicts on full historical transform
                        # This outputs the full saturated history
                        output = effect.predict(first_transform, predicted_effects)
                    elif isinstance(effect, BaseAdstockEffect):
                        # Adstock receives full saturated history from first effect
                        # Pass as (data, indices) tuple for correct carryover + selection
                        adstock_input = (output, horizon_indices)
                        output = effect.predict(adstock_input, predicted_effects)
                    else:
                        # Non-adstock effect: use output from previous
                        output = effect.predict(output, predicted_effects)
            return output

        # Simple format: backwards compatible
        output = data
        for name, effect in self.named_steps_:
            with numpyro.handlers.scope(prefix=name):
                output = effect.predict(output, predicted_effects)
        return output

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from prophetverse.effects.linear import LinearEffect
        from prophetverse.effects.adstock import GeometricAdstockEffect

        return [
            {
                "steps": [
                    ("adstock", GeometricAdstockEffect()),
                    ("linear", LinearEffect(effect_mode="multiplicative")),
                ]
            },
            {
                "steps": [
                    ("linear", LinearEffect(effect_mode="additive")),
                    ("adstock", GeometricAdstockEffect()),
                ]
            },
        ]
