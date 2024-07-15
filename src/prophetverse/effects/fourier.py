"""Fourier effects for time series forecasting with seasonality."""

from typing import Dict, List, Union

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseEffect, Stage
from prophetverse.effects.linear import LinearEffect
from prophetverse.sktime._expand_column_per_level import ExpandColumnPerLevel

__all__ = ["LinearFourierSeasonality"]


class LinearFourierSeasonality(BaseEffect):
    """Linear Fourier Seasonality effect.

    Compute the linear seasonality using Fourier features.

    Parameters
    ----------
    sp_list : List[float]
        List of seasonal periods.
    fourier_terms_list : List[int]
        List of number of Fourier terms to use for each seasonal period.
    freq : str
        Frequency of the time series. Example: "D" for daily, "W" for weekly, etc.
    prior_scale : float, optional
        Scale of the prior distribution for the effect, by default 1.0.
    effect_mode : str, optional
        Either "multiplicative" or "additive" by default "additive".
    """

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "supports_multivariate": True,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": False,
    }

    def __init__(
        self,
        sp_list: List[float],
        fourier_terms_list: List[int],
        freq: Union[str, None],
        prior_scale: float = 1.0,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive",
    ):
        self.sp_list = sp_list
        self.fourier_terms_list = fourier_terms_list
        self.freq = freq
        self.prior_scale = prior_scale
        self.effect_mode = effect_mode
        self.expand_column_per_level_ = None  # type: Union[None,ExpandColumnPerLevel]

    def _fit(self, X: pd.DataFrame, scale: float = 1.0):
        """Customize the initialization of the effect.

        Fit the fourier feature transformer and the linear effect.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to initialize the effect.
        scale: float, optional
            The scale of the timeseries, by default 1.0.
        """
        self.fourier_features_ = FourierFeatures(
            sp_list=self.sp_list,
            fourier_terms_list=self.fourier_terms_list,
            freq=self.freq,
            keep_original_columns=True,
        )

        self.fourier_features_.fit(X=X)
        X = self.fourier_features_.transform(X)

        if X.index.nlevels > 1 and X.index.droplevel(-1).nunique() > 1:
            self.expand_column_per_level_ = ExpandColumnPerLevel([".*"]).fit(X=X)
            X = self.expand_column_per_level_.transform(X)  # type: ignore

        self.linear_effect_ = LinearEffect(
            prior=dist.Normal(0, self.prior_scale), effect_mode=self.effect_mode
        )

        self.linear_effect_.fit(X=X, scale=scale)

    def _transform(
        self, X: pd.DataFrame, stage: Stage = Stage.TRAIN
    ) -> Dict[str, jnp.ndarray]:
        """Prepare the input data in a dict of jax arrays.

        Creates the fourier terms and the linear effect.

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
            A dictionary containing the data needed for the effect.
        """
        X = self.fourier_features_.transform(X)

        if self.expand_column_per_level_ is not None:
            X = self.expand_column_per_level_.transform(X)

        array = self.linear_effect_.transform(X, stage)

        return array

    def _predict(self, trend: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Apply the effect.

        Apply linear seasonality.

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
        return self.linear_effect_.predict(trend, **kwargs)
