"""Definition of Chained Effects class."""

from typing import Any, Dict, List

import jax.numpy as jnp
from numpyro import handlers
from skbase.base import BaseMetaEstimatorMixin

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
        "supports_multivariate": True,
        "skip_predict_if_no_match": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(self, steps: List[BaseEffect]):
        self.steps = steps
        super().__init__()

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
        for effect in self.steps:
            effect.fit(y, X, scale)

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
        output = self.steps[0].transform(output, fh)
        return output

    def _sample_params(
        self, data: jnp.ndarray, predicted_effects: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample parameters for all chained effects.

        Parameters
        ----------
        data : jnp.ndarray
            Data obtained from the transformed method.
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters for all effects.
        """
        params = {}
        for idx, effect in enumerate(self.steps):
            with handlers.scope(prefix=f"{idx}"):
                effect_params = effect.sample_params(data, predicted_effects)
            params[f"effect_{idx}"] = effect_params
        return params

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        params: Dict[str, Dict[str, jnp.ndarray]],
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
        for idx, effect in enumerate(self.steps):
            effect_params = params[f"effect_{idx}"]
            output = effect._predict(output, predicted_effects, effect_params)
        return output

    def _coerce_to_named_object_tuples(self, objs, clone=False, make_unique=True):
        """Coerce sequence of objects or named objects to list of (str, obj) tuples.

        Input that is sequence of objects, list of (str, obj) tuples or
        dict[str, object] will be coerced to list of (str, obj) tuples on return.

        Parameters
        ----------
        objs : list of objects, list of (str, object tuples) or dict[str, object]
            The input should be coerced to list of (str, object) tuples. Should
            be a sequence of objects, or follow named object API.
        clone : bool, default=False.
            Whether objects in the returned list of (str, object) tuples are
            cloned (True) or references (False).
        make_unique : bool, default=True
            Whether the str names in the returned list of (str, object) tuples
            should be coerced to unique str values (if str names in input
            are already unique they will not be changed).

        Returns
        -------
        list[tuple[str, BaseObject]]
            List of tuples following named object API.

            - If `objs` was already a list of (str, object) tuples then either the
              same named objects (as with other cases cloned versions are
              returned if ``clone=True``).
            - If `objs` was a dict[str, object] then the named objects are unpacked
              into a list of (str, object) tuples.
            - If `objs` was a list of objects then string names were generated based
               on the object's class names (with coercion to unique strings if
               necessary).
        """
        objs = [(f"effect_{idx}", obj) for idx, obj in enumerate(objs)]
        return super()._coerce_to_named_object_tuples(objs, clone, make_unique)
