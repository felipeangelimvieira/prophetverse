"""Extension template for creating a new effect."""

from typing import Any, Dict

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect, Stage
from prophetverse.utils.frame_to_array import series_to_tensor_or_array


class MyEffectName(BaseEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "supports_multivariate": False,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": True,
    }

    def __init__(self, param1: Any, param2: Any):
        self.param1 = param1
        self.param2 = param2

    def _fit(self, X: pd.DataFrame, scale: float = 1.0):
        """Customize the initialization of the effect.

        This method is called by the `fit()` method and can be overridden by
        subclasses to provide additional initialization logic.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to initialize the effect.
        """
        # Do something with X, scale, and other parameters
        pass

    def _transform(
        self, X: pd.DataFrame, stage: Stage = Stage.TRAIN
    ) -> Dict[str, jnp.ndarray]:
        """Prepare the input data in a dict of jax arrays.

        This method is called by the `fit()` method and can be overridden
        by subclasses to provide additional data preparation logic.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the exogenous variables for the training
            time indexes, if passed during fit, or for the forecasting time indexes, if
            passed during predict.

        stage : Stage, optional
            The stage of the effect, by default Stage.TRAIN. This can be used to
            differentiate between training and prediction stages and apply different
            transformations accordingly.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the data needed for the effect. The keys of the
            dictionary should be the names of the arguments of the `apply` method, and
            the values should be the corresponding data as jnp.ndarray.
        """
        # Do something with X
        if stage == "train":
            array = series_to_tensor_or_array(X)
        else:
            # something else
            pass
        return {"data": array}

    def _predict(self, trend: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Apply the effect.

        This method is called by the `apply()` method and must be overridden by
        subclasses to provide the actual effect computation logic.

        Parameters
        ----------
        trend : jnp.ndarray
            An array containing the trend values.

        kwargs: dict
            Additional keyword arguments that may be needed to compute the effect.

        Returns
        -------
        jnp.ndarray
            The effect values.
        """
        raise NotImplementedError("Subclasses must implement _predict()")
