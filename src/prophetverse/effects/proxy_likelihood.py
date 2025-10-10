"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect

__all__ = ["ProxyLikelihood"]


class LinearProxyLikelihood(BaseEffect):
    """
    Use a proxy variable as a likelihood for an effect.

    Assumes there exist an observed variable Z that is correlated with certain
    components we want to predict. This effect adds the likelihood:

    $$
    Z_t \sim \mathcal{N}(\beta f_i, \sigma)
    $$

    Parameters
    ----------
    effect_names : str
        The effect to use in the likelihood.
    reference_df : pd.DataFrame
        A dataframe with the proxy values. Should be in sktime format, and must
        have the same index as the input data.
    coefficient_prior : numpyro.distributions.Distribution, optional
        Prior distribution for the coefficient beta. If None,
          a Normal(1, 0.5) prior is used.
    likelihood_scale : float
        The scale of the likelihood

    """

    _tags = {"requires_X": False, "hierarchical_prophet_compliant": False}

    def __init__(
        self,
        effect_name: str,
        reference_df: pd.DataFrame,
        coefficient_prior=None,
        likelihood_scale: float = 0.05,
        should_rescale: bool = False,
    ):

        self.effect_name = effect_name
        self.reference_df = reference_df
        self.likelihood_scale = likelihood_scale
        self.coefficient_prior = coefficient_prior
        self.should_rescale = should_rescale
        super().__init__()

        assert self.likelihood_scale > 0, "likelihood_scale must be greater than 0"

        self._coefficient_prior = coefficient_prior
        if coefficient_prior is None:
            self._coefficient_prior = dist.Normal(1.0, 0.5)

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.


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
        self.scale_ = scale

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

        X_ref = self.reference_df.reindex(fh, fill_value=jnp.nan)
        lift_array = series_to_tensor_or_array(X_ref)
        data_dict["observed_reference_value"] = lift_array
        if self.should_rescale:
            data_dict["observed_reference_value"] /= self.scale_
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

        beta = numpyro.sample("beta", self._coefficient_prior)

        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "proxy_likelihood:ignore",
                dist.Normal(x * beta, self.likelihood_scale),
                obs=observed_reference_value,
            )

        return jnp.zeros_like(x)

    @classmethod
    def get_test_params(cls, parameter_set="default"):

        return [
            {
                "effect_name": "trend",
                "reference_df": pd.DataFrame({"y": [1, 2, 3]}),
                "likelihood_scale": 0.1,
            }
        ]
