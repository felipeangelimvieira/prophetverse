"""Univariate Prophet model.

This module implements the Univariate Prophet model, similar to the one implemented in
the `prophet` library.
"""

import warnings
from typing import List, Optional, Union, Dict

import numpyro
import jax.numpy as jnp
import jax
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
from prophetverse.utils import series_to_tensor, reindex_time_series
from collections import defaultdict
from prophetverse.utils import get_multiindex_loc
from prophetverse.utils.frame_to_array import series_to_tensor

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
        broadcast_mode="estimator",
    ):
        """Initialize the Prophetverse model."""
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.likelihood = likelihood
        self.broadcast_mode = broadcast_mode

        super().__init__(
            rng_key=rng_key,
            trend=trend,
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
            inference_engine=inference_engine,
            scale=scale,
        )

        self._validate_hyperparams()

        if self.broadcast_mode != "estimator":
            self.set_tags(
                **{
                    "y_inner_mtype": [
                        "pd.DataFrame",
                        "pd-multiindex",
                        "pd_multiindex_hier",
                    ],
                    "X_inner_mtype": [
                        "pd.DataFrame",
                        "pd-multiindex",
                        "pd_multiindex_hier",
                    ],
                }
            )

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

        if not self.broadcast_mode in ["estimator", "effect"]:
            raise ValueError(
                f"broadcast_mode must be either 'estimator' or 'effect'. Got '{self.broadcast_mode}'."
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
        if X is None:
            X = pd.DataFrame(index=y.index)
        self.trend_model_ = self._trend.clone()
        self.likelihood_model_ = self._likelihood.clone()

        if self._likelihood_is_discrete:
            # Scale the data for discrete likelihoods to avoid non-integer values.
            self.trend_model_.fit(X=X, y=y / self._scale, scale=1)
            self.likelihood_model_.fit(X=X, y=y, scale=self._scale)
        else:
            self.trend_model_.fit(X=X, y=y, scale=self._scale)
            self.likelihood_model_.fit(X=X, y=y, scale=self._scale)

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

        if y.index.nlevels > 1:
            # Panel data
            y_array = series_to_tensor(y)
        else:
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
            idx = reindex_time_series(self._y, fh_as_index).index
            X = pd.DataFrame(index=idx)

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

    def _get_predictive_samples_dict(self, fh, X=None):
        samples = super()._get_predictive_samples_dict(fh, X)
        panel_samples = {k: v for k, v in samples.items() if k.startswith("panel-")}
        for key in panel_samples:
            del samples[key]
        samples.update(group_by_suffix(panel_samples))
        return samples

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
        from prophetverse.engine import MCMCInferenceEngine, MAPInferenceEngine, prior
        from prophetverse.engine.optimizer import AdamOptimizer

        params = [
            {
                "trend": FlatTrend(),
                "inference_engine": MAPInferenceEngine(
                    optimizer=AdamOptimizer(step_size=0.01), num_steps=1
                ),
            },
            {
                "inference_engine": MCMCInferenceEngine(
                    num_chains=1, num_samples=1, num_warmup=1
                ),
                "trend": FlatTrend(),
            },
            {
                "trend": FlatTrend(),
                "broadcast_mode": "effect",
                "inference_engine": MAPInferenceEngine(
                    optimizer=AdamOptimizer(step_size=0.01), num_steps=1
                ),
            },
        ]

        return params

    def _optimizer_predictive_callable(self, X, horizon, columns):
        """
        Return predictive callable for budget optimization.

        Budget optimization requires calling predictive function
        with new values for the exogenous effects, more specifically,
        specific colummns and specific horizons.

        This function returns a callable that can be used
        to predict the model's output given new values for the
        (horizon,columns) pair.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the exogenous variables.
        horizon : pd.Index
            Index of the forecast horizon.
        columns : List[str]
            List of columns to be used in the prediction.

        Returns
        -------
        Callable
            A callable that takes new values for the exogenous effects
            and returns the predicted values for the model.
        """

        model = self

        fh: pd.Index = X.index.get_level_values(-1).unique()
        X = X.copy()

        predict_data = model.get_predict_data(X=X, fh=fh)
        inference_engine = model.inference_engine_

        # Get the indexes of `horizon` in fh

        horizon_idx = jnp.array(X.index.get_level_values(-1).isin(horizon))

        # Prepare exogenous effects -
        # we need to transform them on every call to check the
        # objective function and gradient.
        # We save a triplet (effect_name, effect, effect_columns)
        # where effect_columns is a set of indexes of the columns
        # in the `columns` list that are used by the effect.
        exogenous_effects_to_update = []
        for effect_name, effect, effect_columns in model.exogenous_effects_:
            # If no columns are found, skip
            if effect_columns is None or len(effect_columns) == 0:
                continue

            intersection = effect_columns.intersection(columns)
            if len(intersection) == 0:
                continue

            exogenous_effects_to_update.append(
                (
                    effect_name,
                    effect,
                    # index of effect_columns in columns
                    [columns.index(col) for col in intersection],
                )
            )

        x_array = series_to_tensor(X)

        if not isinstance(self._scale, (pd.Series, pd.DataFrame)):
            scale = self._scale
        else:
            scale = self._scale.values.reshape((-1, 1, 1))

        def predictive(new_x):
            """
            Update predict data and call self._predict
            """
            if x_array.ndim <= 2:
                new_x = new_x.reshape(len(horizon), len(columns))
            else:
                new_x = new_x.reshape((x_array.shape[0], len(horizon), len(columns)))
            for effect_name, effect, effect_column_idx in exogenous_effects_to_update:
                _data = x_array[..., effect_column_idx]
                shape = _data.shape
                _data = _data.flatten()
                _data = _data.at[horizon_idx].set(
                    new_x[..., effect_column_idx].flatten()
                )
                _data = _data.reshape(shape)
                # Update the effect data
                predict_data["data"][effect_name] = effect._update_data(
                    predict_data["data"][effect_name], _data
                )

            predictive_samples = inference_engine.predict(**predict_data)
            panel_samples = {
                k: v for k, v in predictive_samples.items() if k.startswith("panel-")
            }
            predictive_samples.update(group_by_suffix(panel_samples))
            obs = predictive_samples["obs"]

            # TODO: there may be a better place to place this
            # This is a workaround
            # if obs.ndim == 4:
            #    obs = obs.squeeze(0)
            obs = obs * scale

            return obs

        return predictive

    def optimize_predictive_callable(self, X, horizon, columns):

        if not self._is_vectorized:
            return self._optimizer_predictive_callable(
                X=X, horizon=horizon, columns=columns
            )

        callables = []
        for idx, data in self.forecasters_.iterrows():
            forecaster = data[0]

            if X is None:
                _X = None
            else:
                _X = get_multiindex_loc(X, [idx])
                # Keep only index level -1
                for _ in range(_X.index.nlevels - 1):
                    _X = _X.droplevel(0)

            callable = forecaster.optimize_predictive_callable(
                X=_X,
                horizon=horizon,
                columns=columns,
            )

            callables.append(callable)

        def broadcasted_callable(new_x):
            new_x = new_x.reshape((-1, len(horizon), len(columns)))
            outs = []
            for i in range(new_x.shape[0]):
                callable = callables[i]
                out = callable(new_x[i].flatten())
                outs.append(out)

            # Swap samples and panels dimensions

            # out can be of shape (Samples, time, 1)
            # or (1, Samples, time, 1)
            if outs[0].ndim == 3:
                outs = jnp.stack(outs, axis=0)
            else:
                outs = jnp.concatenate(outs, axis=0)
            out = outs.transpose(1, 0, 2, 3)
            return out

        return broadcasted_callable


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


def group_by_suffix(data_dict):
    """
    Given a dict whose keys are like "panel-<i>/<suffix>",
    returns a new dict mapping each suffix to a list of values
    ordered by the panel index i.
    """
    temp = defaultdict(list)
    for key, value in data_dict.items():
        try:
            split = key.split("/", 1)
            panel, suffix = split[0], split[-1]
            # extract the integer index from "panel-<i>"
            idx = int(panel.split("-", 1)[1])
        except (ValueError, IndexError) as e:
            # skip keys that don’t match the expected pattern
            warnings.warn(
                f"Key '{key}' does not match expected pattern 'panel-<i>/<suffix>': {e}"
            )
            continue
        temp[suffix].append((idx, value))

    # sort each list by index and strip off the index
    return {
        suffix: jnp.stack([val for idx, val in sorted(lst, key=lambda x: x[0])], axis=1)
        for suffix, lst in temp.items()
    }
