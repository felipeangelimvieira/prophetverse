"""Chained effects utilities.

Contains a shared base class for chaining multiple effects and two concrete
implementations: `Sum` (additive chaining) and `MultiplyEffects` (multiplicative
chaining). This consolidates duplicated logic into a single file.
"""

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
from skbase.base import BaseMetaEstimatorMixin
import numpyro
from prophetverse.effects.base import BaseEffect

__all__ = ["Sum", "MultiplyEffects"]


class _BaseChainOperation(BaseMetaEstimatorMixin, BaseEffect):
    """Base class with shared logic for chaining multiple effects.

    Subclasses must implement `_predict` to define how per-effect
    predictions are aggregated (e.g., sum or product).
    """

    _tags = {
        # Can handle panel data?
        "capability:panel": True,
        # Can be used with hierarchical Prophet?
        "capability:panel": True,
        # Can handle multiple input feature columns?
        "capability:multivariate_input": True,
        # If no columns are found, should
        # _predict be skipped?
        "requires_X": False,
        # Should only the indexes related to the forecasting horizon be passed to
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "named_object_parameters": "named_effects",
    }

    def __init__(self, effects: List[Tuple[str, BaseEffect]]):
        self.effects = effects
        super().__init__()

        # Normalize to list of (name, effect)
        self.named_effects = []
        for i, val in enumerate(self.effects):
            if isinstance(val, tuple):
                self.named_effects.append(val)
            elif isinstance(val, BaseEffect):
                self.named_effects.append((str(i), val))
            else:  # pragma: no cover
                raise ValueError(
                    f"Invalid type {type(val)} for step {i}. Must be a tuple or BaseEffect."
                )

        requires_X = any(
            effect.get_tag("requires_X", False) for _, effect in self.named_effects
        )
        self.set_tags(requires_X=requires_X)

    def _fit(self, y: Any, X: Any, scale: float = 1.0):
        """Fit all chained effects sequentially."""
        self.named_effects_ = []
        for name, effect in self.named_effects:
            effect = effect.clone()
            effect.fit(y, X, scale)
            self.named_effects_.append((name, effect))

    def _transform(self, X: Any, fh: Any) -> Any:
        """Transform input data sequentially through all chained effects."""
        all_data = {
            name: effect.transform(X=X, fh=fh) for name, effect in self.named_effects_
        }
        return all_data

    def _update_data(self, data, arr):
        for name, effect in self.named_effects_:
            data[name] = effect._update_data(data[name], arr)
        return data

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from prophetverse.effects.linear import LinearEffect

        return [
            {
                "effects": [
                    ("linear1", LinearEffect()),
                    ("linear2", LinearEffect()),
                ]
            },
            {"effects": [LinearEffect()]},
        ]

    def _predict(
        self, *args, **kwargs
    ) -> jnp.ndarray:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError()


class MultiplyEffects(_BaseChainOperation):
    """Chains multiple effects multiplicatively."""

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        output = 1
        for name, effect in self.named_effects_:
            with numpyro.handlers.scope(prefix=name):
                output *= effect.predict(data[name], predicted_effects)

        return output


class SumEffects(_BaseChainOperation):
    """Chains multiple effects additively (sums their contributions)."""

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        output = 0
        for name, effect in self.named_effects_:
            with numpyro.handlers.scope(prefix=name):
                output = output + effect.predict(data[name], predicted_effects)

        return output
