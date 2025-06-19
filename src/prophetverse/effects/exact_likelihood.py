"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect

__all__ = ["ExactLikelihood"]


class ExactLikelihood(BaseEffect):
    """Wrap an effect and applies a normal likelihood to its output.

    This class uses an input as a reference for the effect, and applies a normal
    likelihood to the output of the effect.

    Parameters
    ----------
    effect_name : str
        The effect to use in the likelihood.
    reference_df : pd.DataFrame
        A dataframe with the reference values. Should be in sktime format, and must
        have the same index as the input data.
    prior_scale : float
        The scale of the prior distribution for the likelihood.
    """

    _tags = {"requires_X": False, "hierarchical_prophet_compliant": False}

    def __init__(
        self,
        effect_name: str,
        reference_df: pd.DataFrame,
        prior_scale: float,
    ):

        self.effect_name = effect_name
        self.reference_df = reference_df
        self.prior_scale = prior_scale

        assert self.prior_scale > 0, "prior_scale must be greater than 0"

        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.

        This method is called during `fit()` of the forecasting model.
        It receives the Exogenous variables DataFrame and should be used to initialize
        any necessary parameters or data structures, such as detecting the columns that
        match the regex pattern.

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
        self.timeseries_scale = scale

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

        X_lift = self.reference_df.reindex(fh, fill_value=jnp.nan)
        lift_array = series_to_tensor_or_array(X_lift)
        data_dict["observed_reference_value"] = lift_array / self.timeseries_scale
        data_dict["obs_mask"] = ~jnp.isnan(data_dict["observed_reference_value"])
        data_dict["data"] = None
        return data_dict

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
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
        observed_reference_value = data["observed_reference_value"]
        obs_mask = data["obs_mask"]

        x = predicted_effects[self.effect_name]

        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "exact_likelihood:ignore",
                dist.Normal(x, self.prior_scale),
                obs=observed_reference_value,
            )

        return x

    @classmethod
    def get_test_params(cls, parameter_set="default"):

        return [
            {
                "effect_name": "trend",
                "reference_df": pd.DataFrame({"y": [1, 2, 3]}),
                "prior_scale": 0.1,
            }
        ]
