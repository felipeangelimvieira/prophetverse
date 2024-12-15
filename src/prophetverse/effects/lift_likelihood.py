"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import pandas as pd

from prophetverse.distributions import GammaReparametrized
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
        likelihood_scale: float = 1,
    ):

        self.effect = effect
        self.lift_test_results = lift_test_results
        self.prior_scale = prior_scale
        self.likelihood_scale = likelihood_scale

        super().__init__()

        assert self.prior_scale > 0, "prior_scale must be greater than 0"

        mandatory_columns = ["x_start", "x_end", "lift"]
        assert all(
            column in self.lift_test_results.columns for column in mandatory_columns
        ), f"lift_test_results must have the following columns: {mandatory_columns}"

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

        data_dict["observed_lift"] = (
            series_to_tensor_or_array(X_lift["lift"].dropna()) / self.timeseries_scale
        )
        data_dict["x_start"] = series_to_tensor_or_array(X_lift["x_start"].dropna())
        data_dict["x_end"] = series_to_tensor_or_array(X_lift["x_end"].dropna())
        data_dict["obs_mask"] = ~jnp.isnan(series_to_tensor_or_array(X_lift["lift"]))

        return data_dict

    def _sample_params(self, data, predicted_effects=None):
        return self.effect.sample_params(
            data=data["inner_effect_data"], predicted_effects=predicted_effects
        )

    def _predict(
        self,
        data: Dict,
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

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries.
        """
        observed_lift = data["observed_lift"].reshape((-1, 1))
        x_start = data["x_start"].reshape((-1, 1))
        x_end = data["x_end"].reshape((-1, 1))
        obs_mask = data["obs_mask"]

        effect_params = self.effect.sample_params(
            data=data["inner_effect_data"],
            predicted_effects=predicted_effects,
        )

        predicted_effects_masked = {
            k: v[obs_mask] for k, v in predicted_effects.items()
        }

        x = self.effect.predict(
            data=data["inner_effect_data"],
            predicted_effects=predicted_effects,
            params=params,
        )

        y_start = self.effect.predict(
            data=x_start,
            predicted_effects=predicted_effects_masked,
            params=effect_params,
        )
        y_end = self.effect.predict(
            data=x_end, predicted_effects=predicted_effects_masked, params=effect_params
        )

        delta_y = jnp.abs(y_end - y_start)

        with numpyro.handlers.scale(scale=self.likelihood_scale):
            distribution = GammaReparametrized(delta_y, self.prior_scale)

            numpyro.sample(
                "lift_experiment:ignore",
                distribution,
                obs=observed_lift,
            )

        return x
