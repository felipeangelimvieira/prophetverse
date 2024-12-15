"""Fourier effects for time series forecasting with seasonality."""

from typing import Dict, List, Union

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseEffect
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

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Customize the initialization of the effect.

        Fit the fourier feature transformer and the linear effect.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale: float, optional
            The scale of the timeseries, by default 1.0.
        """
        self.fourier_features_ = FourierFeatures(
            sp_list=self.sp_list,
            fourier_terms_list=self.fourier_terms_list,
            freq=self.freq,
            keep_original_columns=False,
        )

        self.fourier_features_.fit(X=X)
        X = self.fourier_features_.transform(X)

        if X.index.nlevels > 1 and X.index.droplevel(-1).nunique() > 1:
            self.expand_column_per_level_ = ExpandColumnPerLevel([".*"]).fit(X=X)
            X = self.expand_column_per_level_.transform(X)  # type: ignore

        self.linear_effect_ = LinearEffect(
            prior=dist.Normal(0, self.prior_scale), effect_mode=self.effect_mode
        )

        self.linear_effect_.fit(X=X, y=y, scale=scale)

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> jnp.ndarray:
        """Prepare input data to be passed to numpyro model.

        This method return a jnp.ndarray of sines and cosines of the given
        frequencies.

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
        jnp.ndarray
            Any object containing the data needed for the effect. The object will be
            passed to `predict` method as `data` argument.
        """
        X = self.fourier_features_.transform(X)

        if self.expand_column_per_level_ is not None:
            X = self.expand_column_per_level_.transform(X)

        array = self.linear_effect_.transform(X, fh)

        return array

    def _sample_params(self, data, predicted_effects=None):
        """Sample parameters from the prior distribution.

        Parameters
        ----------
        data : jnp.ndarray
            The data to be used for sampling the parameters.

        Returns
        -------
        dict
            A dictionary containing the sampled parameters.
        """
        return self.linear_effect_.sample_params(data, predicted_effects)

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
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        return self.linear_effect_.predict(
            data=data,
            predicted_effects=predicted_effects,
            params=params,
        )
