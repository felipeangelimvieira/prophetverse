"""Definition of Chained Effects class."""

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
from skbase.base import BaseMetaEstimatorMixin
import numpyro
from prophetverse.effects.base import BaseEffect

__all__ = ["ChainedEffects"]


class ChainedEffects(BaseMetaEstimatorMixin, BaseEffect):
    """
    Chains multiple effects sequentially, applying them one after the other.

    Parameters
    ----------
    steps : List[BaseEffect]
        A list of effects to be applied sequentially.
    """

    _tags = {
        "hierarchical_prophet_compliant": True,
        "requires_X": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "named_object_parameters": "named_steps",
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
        self.named_steps_ = []
        for name, effect in self.named_steps:
            effect = effect.clone()
            effect.fit(y, X, scale)
            self.named_steps_.append((name, effect))

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
        output = X
        output = self.named_steps_[0][1].transform(output, fh)
        return output

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
                    ("linear", LinearEffect()),
                ]
            },
            {
                "steps": [
                    ("linear", LinearEffect()),
                    ("adstock", GeometricAdstockEffect()),
                ]
            },
        ]
