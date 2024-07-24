"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict, List

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect

__all__ = ["LiftExperimentLikelihood"]


class LiftExperimentLikelihood(BaseEffect):
    """Wrap an effect and applies a normal likelihood to its output.

    This class uses an input as a reference for the effect, and applies a normal
    likelihood to the output of the effect.

    Parameters
    ----------
    effect : BaseEffect
        The effect to wrap.
    lift_test_results : pd.DataFrame
        A dataframe with the lift test results. Should be in sktime format, and must
        have the same index as the input data.
    prior_scale : float
        The scale of the prior distribution for the likelihood.
    """

    _tags = {"skip_predict_if_no_match": False, "supports_multivariate": False}

    def __init__(
        self,
        effect: BaseEffect,
        lift_test_results: pd.DataFrame,
        prior_scale: float,
    ):

        self.effect = effect
        self.lift_test_results = lift_test_results
        self.prior_scale = prior_scale

        assert self.prior_scale > 0, "prior_scale must be greater than 0"

        super().__init__()

    def fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
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
        """
        self.effect.fit(X=X, y=y, scale=scale)
        self.timeseries_scale = scale
        super().fit(X=X, y=y, scale=scale)

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict[str, Any]:
        """Prepare input data to be passed to numpyro model.

        Returns a dictionary with the data for the lift and for the inner effect.

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
        Dict[str, Any]
            Dictionary with data for the lift and for the inner effect
        """
        data_dict = {}
        data_dict["inner_effect_data"] = self.effect._transform(X, fh=fh)

        X_lift = self.lift_test_results.reindex(fh, fill_value=jnp.nan)
        lift_array = series_to_tensor_or_array(X_lift)
        data_dict["observed_lift"] = lift_array / self.timeseries_scale
        data_dict["obs_mask"] = ~jnp.isnan(data_dict["observed_lift"])

        return data_dict

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray]
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
            An array with shape (T,1) for univariate timeseries.
        """
        observed_lift = data["observed_lift"]
        obs_mask = data["obs_mask"]

        x = self.effect.predict(
            data=data["inner_effect_data"], predicted_effects=predicted_effects
        )

        numpyro.sample(
            "lift_experiment",
            dist.Normal(x, self.prior_scale),
            obs=observed_lift,
            obs_mask=obs_mask,
        )

        return x

    @property
    def input_feature_column_names(self) -> List[str]:
        """Return the input feature columns names."""
        return self.effect._input_feature_column_names
