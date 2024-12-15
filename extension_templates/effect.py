"""Extension template for creating a new effect."""

from typing import Any, Dict

import jax.numpy as jnp
import pandas as pd

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array


class MySimpleEffectName(BaseEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "supports_multivariate": False,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": True,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(self, param1: Any, param2: Any):
        self.param1 = param1
        self.param2 = param2

    def _sample_params(self, data, predicted_effects):
        # call numpyro.sample to sample the parameters of the effect
        # return a dictionary with the sampled parameters, where
        # key is the name of the parameter and value is the sampled value
        return {}

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        params : Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        # predicted effects come with the following shapes:
        # (T, 1) shaped array for univariate timeseries
        # (N, T, 1) shaped array for multivariate timeseries, where N is the number of
        # series

        # Here you use the params to compute the effect.
        raise NotImplementedError("Subclasses must implement _predict()")


class MyEffectName(BaseEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "supports_multivariate": False,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": True,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(self, param1: Any, param2: Any):
        self.param1 = param1
        self.param2 = param2

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Customize the initialization of the effect.

        This method is called by the `fit()` method and can be overridden by
        subclasses to provide additional initialization logic.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale : float, optional
            The scale of the timeseries. For multivariate timeseries, this is
            a dataframe. For univariate, it is a simple float.
        """
        # Do something with X, scale, and other parameters
        pass

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Any:
        """Prepare input data to be passed to numpyro model.

        This method receives the Exogenous variables DataFrame and should return a
        the data needed for the effect. Those data will be passed to the `predict`
        method as `data` argument.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the exogenous variables for the training
            time indexes, if passed during fit, or for the forecasting time indexes, if
            passed during predict.

        fh : pd.Index
            The forecasting horizon as a pandas Index.

        Returns
        -------
        Any
            Any object containing the data needed for the effect. The object will be
            passed to `predict` method as `data` argument.
        """
        # Do something with X
        array = series_to_tensor_or_array(X)
        return array

    def _sample_params(self, data, predicted_effects):
        # call numpyro.sample to sample the parameters of the effect
        # return a dictionary with the sampled parameters, where
        # key is the name of the parameter and value is the sampled value
        return {}

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        params : Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        # predicted effects come with the following shapes:
        # (T, 1) shaped array for univariate timeseries
        # (N, T, 1) shaped array for multivariate timeseries, where N is the number of
        # series

        # Here you use the params to compute the effect.
        raise NotImplementedError("Subclasses must implement _predict()")
