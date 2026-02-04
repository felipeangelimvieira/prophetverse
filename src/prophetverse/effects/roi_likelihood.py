"""ROI (Return on Investment) likelihood effect."""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect

__all__ = ["ROILikelihood"]


class ROILikelihood(BaseEffect):
    """Apply a normal likelihood to the ROI (Return on Investment) of an effect.

    This effect computes the ROI as the ratio between the sum of the effect's
    contribution and the sum of the input (captured at transform time). A normal
    likelihood is applied to this ratio, centered at a given value with a given scale.

    The ROI is computed as:
        ROI = sum(effect_contribution) / sum(input)

    Parameters
    ----------
    effect_name : str
        The name of the effect to use for computing ROI. This should match
        the key in predicted_effects dictionary.
    roi_mean : float
        The expected (center) value of the ROI. The normal likelihood will be
        centered at this value.
    roi_scale : float
        The scale (standard deviation) of the normal distribution for the ROI.
        Must be greater than 0.

    Examples
    --------
    >>> from prophetverse.effects import ROILikelihood
    >>> roi_effect = ROILikelihood(
    ...     effect_name="marketing_spend",
    ...     roi_mean=2.0,  # Expect $2 return per $1 spent
    ...     roi_scale=0.5,
    ... )
    """

    _tags = {"requires_X": True, "capability:panel": False}

    def __init__(
        self,
        effect_name: str,
        roi_mean: float,
        roi_scale: float,
    ):
        self.effect_name = effect_name
        self.roi_mean = roi_mean
        self.roi_scale = roi_scale

        assert self.roi_scale > 0, "roi_scale must be greater than 0"

        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.

        This method is called during `fit()` of the forecasting model.
        It receives the Exogenous variables DataFrame and should be used to initialize
        any necessary parameters or data structures.

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

        Captures the input values that will be used to compute ROI.

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
            Dictionary with the captured input data for ROI computation.
        """
        data_dict = {}

        # Capture the input values - sum across all columns if multiple
        input_array = series_to_tensor_or_array(X)
        data_dict["input_values"] = input_array
        data_dict["input_sum"] = jnp.sum(input_array)
        data_dict["data"] = None

        return data_dict

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Computes the ROI and applies a normal likelihood centered at the expected
        ROI value.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with zeros (this effect only adds the likelihood constraint).
        """
        input_sum = data["input_sum"]

        # Get the effect contribution
        effect_contribution = predicted_effects[self.effect_name]
        effect_sum = jnp.sum(effect_contribution)

        # Compute ROI as the ratio of effect sum to input sum
        # Add small epsilon to avoid division by zero
        roi = effect_sum / (input_sum + 1e-10)

        # Apply normal likelihood to the ROI
        # Use :ignore suffix so that this sample is removed from output dataframe
        numpyro.sample(
            "roi_likelihood:ignore",
            dist.Normal(self.roi_mean, self.roi_scale),
            obs=roi,
        )

        # Return zeros - this effect only adds a likelihood constraint
        return jnp.zeros_like(effect_contribution)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {
                "effect_name": "trend",
                "roi_mean": 2.0,
                "roi_scale": 0.5,
            }
        ]
