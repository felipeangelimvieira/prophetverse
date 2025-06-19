"""Flat trend model."""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.effects.base import BaseEffect
from prophetverse.distributions import GammaReparametrized

from .base import TrendEffectMixin


class FlatTrend(TrendEffectMixin, BaseEffect):
    """Flat trend model.

    The mean of the target variable is used as the prior location for the trend.

    Parameters
    ----------
    changepoint_prior_scale : float, optional
        The scale of the prior distribution on the trend changepoints. Defaults to 0.1.
    """

    def __init__(self, changepoint_prior_scale: float = 0.1) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.

        Set the prior location for the trend.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.
        """
        self.changepoint_prior_loc = y.mean().values

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> dict:
        """Prepare input data (a constant factor in this case).

        Parameters
        ----------
        idx : pd.PeriodIndex
            the timeseries time indexes

        Returns
        -------
        dict
            dictionary containing the input data for the trend model
        """
        idx = X.index
        return jnp.ones((len(idx), 1))

    def _predict(  # type: ignore[override]
        self, data: jnp.ndarray, predicted_effects: dict, *args, **kwargs
    ) -> jnp.ndarray:
        """Apply the trend.

        Parameters
        ----------
        constant_vector : jnp.ndarray
            A constant vector with the size of the series time indexes

        Returns
        -------
        jnp.ndarray
            The forecasted trend
        """
        # Alias for clarity
        constant_vector = data

        coefficient = numpyro.sample(
            "trend_flat_coefficient",
            GammaReparametrized(
                loc=self.changepoint_prior_loc,
                scale=self.changepoint_prior_scale,
            ),
        )

        return constant_vector * coefficient
