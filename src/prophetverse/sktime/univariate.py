"""Univariate Prophet model.

This module implements the Univariate Prophet model, similar to the one implemented in
the `prophet` library.
"""

from typing import List, Optional, Union, Dict

import numpyro
import jax.numpy as jnp
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from prophetverse.effects import BaseEffect
from prophetverse.sktime.base import BaseProphetForecaster
from prophetverse.effects.target.univariate import (
    NormalTargetLikelihood,
    NegativeBinomialTargetLikelihood,
    GammaTargetLikelihood,
)
from prophetverse.utils.deprecation import deprecation_warning

__all__ = ["Prophetverse", "Prophet", "ProphetGamma", "ProphetNegBinomial"]

_LIKELIHOOD_MODEL_MAP = {
    "normal": NormalTargetLikelihood,
    "gamma": GammaTargetLikelihood,
    "negbinomial": NegativeBinomialTargetLikelihood,
}

_DISCRETE_LIKELIHOODS = ["negbinomial"]


class Prophetverse(BaseProphetForecaster):
    """Univariate Prophetverse forecaster with multiple likelihood options.

    This forecaster implements a univariate model with support for different likelihoods.
    It differs from Facebook's Prophet in several ways:
      - Logistic trend is parametrized differently, inferring capacity from data.
      - Arbitrary sktime transformers can be used (e.g., FourierFeatures or HolidayFeatures).
      - No default weekly or yearly seasonality; these must be provided via the feature_transformer.
      - Uses 'changepoint_interval' instead of 'n_changepoints' for selecting changepoints.
      - Allows for configuring distinct functions for each exogenous variable effect.

    Parameters
    ----------
    trend : Union[str, BaseEffect], optional
        Type of trend to use. Either "linear" (default) or "logistic", or a custom effect object.
    exogenous_effects : Optional[List[BaseEffect]], optional
        List of effect objects defining the exogenous effects.
    default_effect : Optional[BaseEffect], optional
        The default effect for variables without a specified effect.
    feature_transformer : sktime transformer, optional
        Transformer object to generate additional features (e.g., Fourier terms).
    noise_scale : float, optional
        Scale parameter for the observation noise. Must be greater than 0. (default: 0.05)
    likelihood : str, optional
        The likelihood model to use. One of "normal", "gamma", or "negbinomial". (default: "normal")
    scale : optional
        Scaling value inferred from the data.
    rng_key : optional
        A jax.random.PRNGKey instance, or None.
    inference_engine : optional
        An inference engine for running the model.

    Raises
    ------
    ValueError
        If noise_scale is not greater than 0 or an unsupported likelihood is provided.
    """

    _tags = {
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse",
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "enforce_index_type": [pd.Period, pd.DatetimeIndex],
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        trend: Union[BaseEffect, str] = "linear",
        exogenous_effects: Optional[List[BaseEffect]] = None,
        default_effect: Optional[BaseEffect] = None,
        feature_transformer=None,
        noise_scale=None,
        likelihood="normal",
        scale=None,
        rng_key=None,
        inference_engine=None,
    ):
        """Initialize the Prophetverse model."""
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.likelihood = likelihood

        super().__init__(
            rng_key=rng_key,
            trend=trend,
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
            inference_engine=inference_engine,
            scale=scale,
        )

        self._validate_hyperparams()

    @property
    def _likelihood(self):
        """Return the appropriate model function based on the likelihood.

        Returns
        -------
        Callable
            The model function to be used with Numpyro samplers.
        """
        if isinstance(self.likelihood, BaseEffect):
            return self.likelihood
        if not self.likelihood in _LIKELIHOOD_MODEL_MAP:
            raise ValueError(f"Likelihood '{self.likelihood}' is not supported. ")
        likelihood = _LIKELIHOOD_MODEL_MAP[self.likelihood]().clone()
        if self.noise_scale is not None:
            deprecation_warning(
                "noise_scale",
                current_version="0.6.0",
                extra_message="Use the noise_scale parameter in the likelihood instead."
                " You can import the likelihood from prophetverse.effects import NormalTargetLikelihood",
            )
            likelihood.set_params(noise_scale=self.noise_scale)
        return likelihood

    @property
    def _likelihood_is_discrete(self) -> bool:
        """Determine if the likelihood is discrete.

        Returns
        -------
        bool
            True if the likelihood is discrete; False otherwise.
        """
        return self._likelihood in _DISCRETE_LIKELIHOODS or isinstance(
            self._likelihood, NegativeBinomialTargetLikelihood
        )

    def _validate_hyperparams(self):
        """Validate hyperparameters for the model.

        Raises
        ------
        ValueError
            If noise_scale is not greater than 0 or if an unsupported likelihood is provided.
        """
        super()._validate_hyperparams()

        if self.noise_scale is not None and self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")

        valid_likelihood = isinstance(self._likelihood, BaseEffect) or (
            isinstance(self._likelihood, str)
            and self._likelihood in _LIKELIHOOD_MODEL_MAP
        )
        if not valid_likelihood:
            raise ValueError(
                f"likelihood must be one of {list(_LIKELIHOOD_MODEL_MAP.keys())}"
                f"or a base effect instance. Got '{self.likelihood}'."
            )

    def _get_fit_data(self, y, X, fh):
        """Prepare data for fitting the Numpyro model.

        Parameters
        ----------
        y : pd.DataFrame
            Time series data.
        X : pd.DataFrame
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        dict
            Dictionary containing prepared data for model fitting.
        """
        fh = y.index.get_level_values(-1).unique()

        self.trend_model_ = self._trend.clone()
        self.likelihood_model_ = self._likelihood.clone()

        if self._likelihood_is_discrete:
            # Scale the data for discrete likelihoods to avoid non-integer values.
            self.trend_model_.fit(X=X, y=y / self._scale)
            self.likelihood_model_.fit(X=X, y=y / self._scale)
        else:
            self.trend_model_.fit(X=X, y=y)
            self.likelihood_model_.fit(X=X, y=y)

        # Handle exogenous features.
        if X is None:
            X = pd.DataFrame(index=y.index)

        if self.feature_transformer is not None:
            X = self.feature_transformer.fit_transform(X)

        self._has_exogenous = not X.columns.empty
        X = X.loc[y.index]

        trend_data = self.trend_model_.transform(X=X, fh=fh)
        target_data = self.likelihood_model_.transform(X=y, fh=fh)

        self._fit_effects(X, y)
        exogenous_data = self._transform_effects(X, fh=fh)

        y_array = jnp.array(y.values.flatten()).reshape((-1, 1))

        # Data used in both fitting and prediction.
        self.fit_and_predict_data_ = {
            "trend_model": self.trend_model_,
            "target_model": self.likelihood_model_,
            "exogenous_effects": self.non_skipped_exogenous_effect,
        }

        inputs = {
            "y": y_array,
            "data": exogenous_data,
            "trend_data": trend_data,
            "target_data": target_data,
            **self.fit_and_predict_data_,
        }

        return inputs

    def _get_predict_data(
        self, X: Union[pd.DataFrame, None], fh: ForecastingHorizon
    ) -> dict:
        """Prepare data for making predictions with the Numpyro model.

        Parameters
        ----------
        X : pd.DataFrame or None
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        dict
            Dictionary of prepared data for prediction.
        """
        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        if X is None:
            X = pd.DataFrame(index=fh_as_index)

        if self.feature_transformer is not None:
            X = self.feature_transformer.transform(X)

        trend_data = self.trend_model_.transform(X=X, fh=fh_as_index)
        target_data = self.likelihood_model_.transform(X=None, fh=fh_as_index)

        exogenous_data = self._transform_effects(X, fh_as_index)

        return dict(
            y=None,
            data=exogenous_data,
            trend_data=trend_data,
            target_data=target_data,
            **self.fit_and_predict_data_,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # pragma: no cover
        """Return parameters to be used in sktime unit tests.

        Parameters
        ----------
        parameter_set : str, optional
            The parameter set name (currently ignored).

        Returns
        -------
        List[dict[str, int]]
            A list of dictionaries containing test parameters.
        """
        from prophetverse.effects.trend import FlatTrend
        from prophetverse.engine import MCMCInferenceEngine, MAPInferenceEngine
        from prophetverse.engine.optimizer import AdamOptimizer

        params = [
            {
                "trend": FlatTrend(),
                "inference_engine": MAPInferenceEngine(
                    num_steps=1, optimizer=AdamOptimizer()
                ),
            },
            {
                "inference_engine": MCMCInferenceEngine(
                    num_chains=1, num_samples=1, num_warmup=1
                ),
                "trend": FlatTrend(),
            },
        ]

        return params


class Prophet(Prophetverse):
    """Prophet forecaster implemented in Numpyro.

    This forecaster uses a logistic trend and supports custom feature transformers
    for additional seasonality or holiday effects.

    Parameters
    ----------
    feature_transformer : sktime transformer, optional
        Transformer to generate additional features.
    noise_scale : float, optional
        Scale parameter for observation noise. (default: 0.05)
    trend : str, optional
        Type of trend, either "linear" or "logistic". (default: "logistic")
    exogenous_effects : optional
        List of exogenous effect objects.
    default_effect : optional
        Default effect for variables without a specified effect.
    scale : optional
        Scaling factor inferred from data.
    rng_key : optional
        A jax.random.PRNGKey instance, or None.
    """

    def __init__(
        self,
        feature_transformer=None,
        noise_scale=0.05,
        trend="logistic",
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
        inference_engine=None,
    ):
        super().__init__(
            feature_transformer=feature_transformer,
            noise_scale=noise_scale,
            trend=trend,
            exogenous_effects=exogenous_effects,
            likelihood="normal",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
            inference_engine=inference_engine,
        )


class ProphetGamma(Prophetverse):
    """Prophet forecaster with a gamma likelihood.

    Parameters
    ----------
    noise_scale : float, optional
        Scale parameter for observation noise. (default: 0.05)
    trend : str, optional
        Trend type, either "linear" or "logistic". (default: "logistic")
    exogenous_effects : optional
        List of exogenous effect objects.
    default_effect : optional
        Default effect for variables without a specified effect.
    scale : optional
        Scaling factor inferred from data.
    rng_key : optional
        A jax.random.PRNGKey instance, or None.
    """

    def __init__(
        self,
        noise_scale=0.05,
        trend="logistic",
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
        inference_engine=None,
    ):
        super().__init__(
            noise_scale=noise_scale,
            trend=trend,
            exogenous_effects=exogenous_effects,
            likelihood="gamma",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
            inference_engine=inference_engine,
        )


class ProphetNegBinomial(Prophetverse):
    """Prophet forecaster with negative binomial likelihood.

    Parameters
    ----------
    noise_scale : float, optional
        Scale parameter for observation noise. (default: 0.05)
    trend : str, optional
        Trend type, either "linear" or "logistic". (default: "logistic")
    exogenous_effects : optional
        List of exogenous effect objects.
    default_effect : optional
        Default effect for variables without a specified effect.
    scale : optional
        Scaling factor inferred from data.
    rng_key : optional
        A jax.random.PRNGKey instance, or None.
    """

    def __init__(
        self,
        noise_scale=0.05,
        trend="logistic",
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
        inference_engine=None,
    ):
        super().__init__(
            noise_scale=noise_scale,
            trend=trend,
            exogenous_effects=exogenous_effects,
            likelihood="negbinomial",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
            inference_engine=inference_engine,
        )
