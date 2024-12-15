"""Module that stores abstract class of effects."""

from typing import Any, Dict, Literal, Optional

import jax.numpy as jnp
import pandas as pd
from skbase.base import BaseObject

from prophetverse.utils import series_to_tensor_or_array

__all__ = ["BaseEffect", "BaseAdditiveOrMultiplicativeEffect"]


EFFECT_APPLICATION_TYPE = Literal["additive", "multiplicative"]


class BaseEffect(BaseObject):
    """Base class for effects.

    Effects are objects which are responsible for preparing the data and applying
    a specific effect to the forecast. During preparation of the data (which happens in
    `transform` method), the effect receives the exogenous variables dataframe and can
    use them to prepare the jax arrays that will be used at inference time. During
    inference time, the `predict` method is called, and it should output a new component
    to the additive model of Prophetverse.

    Remember that Prophetverse's models are Generalized Additive Models, which are
    composed of many terms summed together to form the final forecast. Each term is
    represented by an effect.

    Children classes should implement the following methods:


    * `_fit` (optional): This method is called during fit() of the forecasting  and
    should be used to initialize any necessary parameters or data structures.
    It receives the exogenous variables dataframe X, the series `y`, and the scale
    factor `scale` that was used to scale the timeseries.

    * `_transform` (optional): This method receives the exogenous variables
    dataframe, and should return an object containing the data needed for the
    effect. This object will be passed to the predict method as `data`. By default
    the columns of the dataframe that match the regex pattern are selected, and the
    result is converted to a `jnp.ndarray`.

    * `_predict` (mandatory): This method receives the output of `_transform` and
    all previously computed effects. It should return the effect values as a
    `jnp.ndarray`


    Parameters
    ----------
    id : str, optional
        The id of the effect, by default "". Used to identify the effect in the model.
    regex : Optional[str], optional
        A regex pattern to match the columns of the exogenous variables dataframe,
        by default None. If None, and _tags["skip_predict_if_no_match"] is True, the
        effect will be skipped if no columns are found.
    effect_mode : EFFECT_APPLICATION_TYPE, optional
        The mode of the effect, either "additive" or "multiplicative", by default
        "multiplicative". If "multiplicative", the effect multiplies the trend values
        before returning them.


    Attributes
    ----------
    should_skip_predict : bool
        If True, the effect should be skipped during prediction. This is determined by
        the `skip_predict_if_no_match` tag and the presence of input feature columns
        names. If the tag is True and there are no input feature columns names, the
        effect should be skipped during prediction.
    """

    _tags = {
        "supports_multivariate": False,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": True,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
        # Is fit() required before calling transform()?
        "requires_fit_before_transform": False,
    }

    def __init__(self):
        self._is_fitted: bool = False

    def fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Initialize the effect.

        This method is called during `fit()` of the forecasting model.
        It receives the Exogenous variables DataFrame and should be used to initialize
        any necessary parameters or data structures, such as detecting the columns that
        match the regex pattern.

        This method MUST set _input_feature_columns_names to a list of column names

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale : float, optional
            The scale of the timeseries. For multivariate timeseries, this is
            a dataframe. For univariate, it is a simple float.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the effect does not support multivariate data and the DataFrame has more
            than one level of index.
        """
        if not self.get_tag("supports_multivariate", False):
            if X is not None and X.index.nlevels > 1:
                raise ValueError(
                    f"The effect {self.__class__.__name__} does not "
                    + "support multivariate data"
                )

        self._fit(y=y, X=X, scale=scale)
        self._is_fitted = True

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
        pass

    def transform(
        self,
        X: pd.DataFrame,
        fh: pd.Index,
    ) -> Any:
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

        Raises
        ------
        ValueError
            If the effect has not been fitted.
        """
        if not self._is_fitted and self.get_tag("requires_fit_before_transform", True):
            raise ValueError("You must call fit() before calling this method")

        if self.get_tag("filter_indexes_with_forecating_horizon_at_transform", True):
            # Filter when index level -1 is in fh
            if X is not None:
                X = X.loc[X.index.get_level_values(-1).isin(fh)]

        return self._transform(X, fh)

    def _transform(
        self,
        X: pd.DataFrame,
        fh: pd.Index,
    ) -> Any:
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
        array = series_to_tensor_or_array(X)
        return array

    def predict(
        self,
        data: Dict,
        predicted_effects: Optional[Dict[str, jnp.ndarray]] = None,
        params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        if predicted_effects is None:
            predicted_effects = {}

        if params is None:
            params = self.sample_params(data, predicted_effects)

        x = self._predict(data, predicted_effects, params)

        return x

    def sample_params(
        self,
        data: Dict,
        predicted_effects: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """Sample parameters from the prior distribution.

        Parameters
        ----------
        data : Dict
            The data to be used for sampling the parameters, obtained from
            `transform` method.

        predicted_effects : Optional[Dict[str, jnp.ndarray]]
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        Dict
            A dictionary containing the sampled parameters.
        """
        if predicted_effects is None:
            predicted_effects = {}

        return self._sample_params(data, predicted_effects)

    def _sample_params(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
    ):
        """Sample parameters from the prior distribution.

        Should be implemented by subclasses to provide the actual sampling logic.

        Parameters
        ----------
        data : Any
            The data to be used for sampling the parameters, obtained from
            `transform` method.
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        Dict
            A dictionary containing the sampled parameters.
        """
        return {}

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], params: Dict
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        raise NotImplementedError("Subclasses must implement _predict()")

    def __call__(
        self,
        data: Dict,
        predicted_effects: Dict[str, jnp.ndarray],
        params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Run the processes to calculate effect as a function."""
        return self.predict(data=data, predicted_effects=predicted_effects)


class BaseAdditiveOrMultiplicativeEffect(BaseEffect):
    """
    Base class for effects that can be applied in additive or multiplicative mode.

    In additive mode, the effect is directly returned. In multiplicative mode,
    the effect is multiplied by the trend before being returned. In other words:

    Additive effect: effect = _predict(trend, **kwargs)
    Multiplicative effect: effect = trend * _predict(trend, **kwargs)

    Parameters
    ----------
    id : str, optional
        The id of the effect, by default "".
    regex : Optional[str], optional
        A regex pattern to match the columns of the exogenous variables dataframe,
        by default None. If None, and _tags["skip_predict_if_no_match"] is True, the
        effect will be skipped if no columns are found.
    effect_mode : EFFECT_APPLICATION_TYPE, optional. Can be "additive"
        or "multiplicative".
    """

    def __init__(self, effect_mode="additive", base_effect_name: str = "trend"):

        self.effect_mode = effect_mode
        self.base_effect_name = base_effect_name

        if effect_mode not in ["additive", "multiplicative"]:
            raise ValueError(
                f"Invalid effect mode: {effect_mode}. "
                + "Effect mode must be 'additive' or 'multiplicative'."
            )

        super().__init__()

    def predict(
        self,
        data: Any,
        predicted_effects: Optional[Dict[str, jnp.ndarray]] = None,
        params: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        if predicted_effects is None:
            predicted_effects = {}

        if params is None:
            params = self.sample_params(data, predicted_effects)

        if (
            self.base_effect_name not in predicted_effects
            and self.effect_mode == "multiplicative"
        ):
            raise ValueError(
                "BaseAdditiveOrMultiplicativeEffect requires trend in"
                + " predicted_effects"
            )

        x = super().predict(
            data=data, predicted_effects=predicted_effects, params=params
        )

        if self.effect_mode == "additive":
            return x

        base_effect = predicted_effects[self.base_effect_name]
        if base_effect.ndim == 1:
            base_effect = base_effect.reshape((-1, 1))
        x = x.reshape(base_effect.shape)
        return base_effect * x
