"""Definition of Chained Effects class."""

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
from skbase.base import BaseMetaEstimatorMixin
import numpyro
from prophetverse.effects.base import BaseEffect

__all__ = ["ChainedEffects"]


class MultiplyEffects(BaseMetaEstimatorMixin, BaseEffect):
    """
    Chains multiple effects sequentially, applying them one after the other.

    Parameters
    ----------
    steps : List[BaseEffect]
        A list of effects to be applied sequentially.
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
        "requires_X": True,
        # Should only the indexes related to the forecasting horizon be passed to
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "named_object_parameters": "effects",
    }

    def __init__(self, effects: List[Tuple[str, BaseEffect]]):
        self.effects = effects
        super().__init__()

        self.named_effects = []
        for i, val in enumerate(self.effects):
            if isinstance(val, tuple):
                self.named_effects.append(val)
            elif isinstance(val, BaseEffect):
                self.named_effects.append((str(i), val))
            else:
                raise ValueError(
                    f"Invalid type {type(val)} for step {i}. Must be a tuple or BaseEffect."
                )

        requires_X = any(
            effect.get_tag("requires_X", False) for _, effect in self.named_effects
        )
        self.set_tags(requires_X=requires_X)

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
        self.named_effects_ = []
        for name, effect in self.named_effects:
            effect = effect.clone()
            effect.fit(y, X, scale)
            self.named_effects_.append((name, effect))

    def _transform(self, X: Any, fh: Any) -> Any:
        """
        Transform input data sequentially through all chained effects.

        Parameters
        ----------
        X : Any
            Input data (e.g., exogenous variables).
        fh : Any
            Forecasting horizon.

        Returns
        -------
        Any
            Transformed data after applying all effects.
        """
        all_data = {
            name: effect.transform(X=X, fh=fh) for name, effect in self.named_effects_
        }
        return all_data

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Apply all chained effects sequentially.

        Parameters
        ----------
        data : jnp.ndarray
            Data obtained from the transformed method (shape: T, 1).
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.
        params : Dict[str, Dict[str, jnp.ndarray]]
            A dictionary containing the sampled parameters for each effect.

        Returns
        -------
        jnp.ndarray
            The transformed data after applying all effects.
        """
        output = 1
        for name, effect in self.named_effects_:
            with numpyro.handlers.scope(prefix=name):
                output *= effect.predict(data[name], predicted_effects)

        return output

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
            {
                "effects": [
                    ("linear1", LinearEffect()),
                ]
            },
        ]

    def _update_data(self, data, arr):
        for name, effect in self.named_effects_:
            data[name] = effect._update_data(data[name], arr)
        return data
