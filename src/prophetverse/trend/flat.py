"""Flat trend model."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .base import TrendModel


class FlatTrend(TrendModel):
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

    def initialize(self, y: pd.DataFrame):
        """Set the prior location for the trend.

        Parameters
        ----------
        y : pd.DataFrame
            The target variable.
        """
        self.changepoint_prior_loc = y.mean().values

    def fit(self, idx: pd.PeriodIndex) -> dict:
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
        return {
            "constant_vector": jnp.ones((len(idx), 1)),
        }

    def compute_trend(  # type: ignore[override]
        self, constant_vector: jnp.ndarray, **kwargs
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
        mean = self.changepoint_prior_loc
        var = self.changepoint_prior_scale**2

        rate = mean / var
        concentration = mean * rate

        coefficient = numpyro.sample(
            "trend_flat_coefficient",
            dist.Gamma(
                rate=rate,
                concentration=concentration,
            ),
        )

        return constant_vector * coefficient
