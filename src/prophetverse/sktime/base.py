import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import sample
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive, init_to_mean
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from collections import OrderedDict
from prophetverse.engine import MAPInferenceEngine, MCMCInferenceEngine, InferenceEngine
from prophetverse.effects import LinearEffect
from prophetverse.utils.frame_to_array import series_to_tensor
import arviz as az
import re
import logging

class BaseBayesianForecaster(BaseForecaster):
    """
    Base class for Bayesian forecasters in hierarchical-prophet.

    Args:
        rng_seed (int): Random number generator seed.
        method (str): Inference method to use. Currently, only "mcmc" is supported.
        num_samples (int): Number of MCMC samples to draw.
        num_warmup (int): Number of warmup steps for MCMC.
        num_chains (int): Number of MCMC chains to run.
        
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    _tags = {
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        rng_key,
        inference_method,
        mcmc_samples,
        mcmc_warmup,
        mcmc_chains,
        optimizer_steps,
        optimizer_name,
        optimizer_kwargs,
        *args,
        **kwargs,
    ):

        self.rng_key = rng_key
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.mcmc_chains = mcmc_chains
        self.inference_method = inference_method
        self.optimizer_steps = optimizer_steps
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self._sample_sites = set()
        super().__init__(*args, **kwargs)
        self.predictive_samples_ = None

    @property
    def optimizer(self):
        if self.optimizer_name.startswith('optax'):

            from numpyro.optim import optax_to_numpyro
            import optax
            scheduler = optax.cosine_decay_schedule(**self.optimizer_kwargs)

            opt = optax_to_numpyro(
                optax.chain(
                    
                    optax.scale_by_adam(),
                    optax.scale_by_schedule(scheduler),
                    optax.scale(-1.0))
            )
            return opt

        return getattr(numpyro.optim, self.optimizer_name)(**self.optimizer_kwargs)

    def _get_fit_data(self, y, X, fh) -> Dict[str, Any]:
        """
        Get the data required for the Numpyro model.

        This method should be implemented by subclasses.

        Args:
            y (pd.DataFrame): Target variable.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            Dict[str, Any]: Data required for the Numpyro model.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def _get_predict_data(self, X, fh):
        """
        Generate samples from the posterior predictive distribution.

        This method should be implemented by subclasses.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            pd.DataFrame: Samples from the posterior predictive distribution.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def model(self, *args, **kwargs):
        """
        Numpyro model.

        This method should be implemented by subclasses.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: Model output.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def _fit(self, y, X, fh):
        """
        Fit the Bayesian forecaster to the training data.

        Args:
            y (pd.DataFrame): Target variable.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            self: The fitted Bayesian forecaster.
        """

        self._set_y_scales(y)
        y  = self._scale_y(y)

        data = self._get_fit_data(y, X, fh)

        self.distributions_ = data.get("distributions", {})

        if self.inference_method == "mcmc":
            self.inference_engine_ = MCMCInferenceEngine(self.model, num_samples=self.mcmc_samples, num_warmup=self.mcmc_warmup, num_chains=self.mcmc_chains, rng_key=self.rng_key)
        elif self.inference_method == "map":
            self.inference_engine_ = MAPInferenceEngine(
                self.model,
                rng_key=self.rng_key,
                optimizer=self.optimizer,
                num_steps=self.optimizer_steps,
            )
        else:
            raise ValueError(f"Unknown method {self.inference_method}")

        self.inference_engine_.infer(**data)
        self.posterior_samples_ = self.inference_engine_.posterior_samples_

        return self

    def _predict(self, fh, X):
        """
        Generate point forecasts for the given forecasting horizon.

        Args:
            fh (ForecastingHorizon): Forecasting horizon.
            X (pd.DataFrame): Exogenous variables.

        Returns:
            pd.DataFrame: Point forecasts for the forecasting horizon.
        """
        predictive_samples = self.predict_samples(fh=fh, X=X)
        self.forecast_samples_ = predictive_samples
        y_pred = predictive_samples.mean(axis=1).to_frame(self._y.columns[0])

        return y_pred

    def predict_all_sites(self, fh, X=None):

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)
            
        fh_as_index = self.fh_to_index(fh)

        predict_data = self._get_predict_data(X=X,fh= fh)

        predictive_samples_ = self.inference_engine_.predict(**predict_data)
        out = pd.DataFrame(
            data={
                site: data.mean(axis=0).flatten()
                for site, data in predictive_samples_.items()
            },
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()
        return self._inv_scale_y(out)

    def predict_samples(
        self,
        fh,
        X=None
):
        """
        Generate samples from the posterior predictive distribution.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            pd.DataFrame: Samples from the posterior predictive distribution.
        """
        fh_as_index = self.fh_to_index(fh)

        predict_data = self._get_predict_data(X=X, fh=fh)

        self.predictive_samples_ = self.inference_engine_.predict(**predict_data)

        observation_site = self.predictive_samples_["obs"]
        n_samples = self.predictive_samples_["obs"].shape[0]
        preds = pd.DataFrame(
            data=observation_site.T.reshape((-1, n_samples)),
            columns=list(range(n_samples)),
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()

        return self._inv_scale_y(preds)

    def _set_y_scales(self, y):
        if y.index.nlevels == 1:
            self._scale = y.abs().max().values[0]
        else:
            self._scale = y.groupby(level=list(range(y.index.nlevels - 1))).agg(lambda x: np.abs(x).max())

    def _scale_y(self, y):
        if y.index.nlevels == 1:
            return y / self._scale
        scale_for_each_obs = self._scale.loc[y.index.droplevel(-1)].values
        return y / scale_for_each_obs

    def _inv_scale_y(self, y):
        if y.index.nlevels == 1:
            return y * self._scale
        scale_for_each_obs = self._scale.loc[y.index.droplevel(-1)].values
        return y * scale_for_each_obs

    def _predict_quantiles(self, fh, X, alpha):
        """
        Generate quantile forecasts for the given forecasting horizon.

        Args:
            fh (ForecastingHorizon): Forecasting horizon.
            X (pd.DataFrame): Exogenous variables.
            alpha (float or List[float]): Quantile(s) to compute.

        Returns:
            pd.DataFrame: Quantile forecasts for the forecasting horizon.
        """
        if isinstance(alpha, float):
            alpha = [alpha]

        forecast_samples_ = self.predict_samples(X=X, fh=fh)

        var_names = self._get_varnames()
        var_name = var_names[0]
        int_idx = pd.MultiIndex.from_product([var_names, alpha])

        quantiles = forecast_samples_.quantile(alpha, axis=1).T
        quantiles.columns = int_idx

        return quantiles

    def periodindex_to_multiindex(self, periodindex: pd.PeriodIndex) -> pd.MultiIndex:
        """
        Convert a PeriodIndex to a MultiIndex.

        Args:
            periodindex (pd.PeriodIndex): PeriodIndex to convert.

        Returns:
            pd.MultiIndex: Converted MultiIndex.
        """
        if self._y.index.nlevels == 1:
            return periodindex

        base_levels = self._y.index.droplevel(-1).unique().tolist()

        # import Iterable
        from collections.abc import Iterable

        # Check if base_levels 0 is a iterable:
        if not isinstance(base_levels[0], tuple):
            base_levels = [(x,) for x in base_levels]

        return pd.Index(
            map(
                lambda x: (*x[0], x[1]),
                itertools.product(base_levels, periodindex),
            ),
            name=self._y.index.names,
        )

    def fh_to_index(self, fh: ForecastingHorizon, y: pd.DataFrame = None):
        """
        Convert a ForecastingHorizon to an index.

        Args:
            fh (ForecastingHorizon): Forecasting horizon.
            y (pd.DataFrame, optional): Target variable. Defaults to None.

        Returns:
            pd.Index: Index corresponding to the forecasting horizon.
        """
        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        if y is None:
            y = self._y
        fh_dates = fh.to_absolute_index(cutoff=y.index.get_level_values(-1).max())
        return fh_dates

    @property
    def var_names(self):

        return list(self.distributions_.keys())

    def plot_trace(self, *, figsize=(15,25), compact=True, var_names=None, **kwargs):

        if var_names is None:
            var_names = self.var_names

        return az.plot_trace(
            az.from_numpyro(self.inference_engine_.mcmc_),
            var_names=var_names,
            filter_vars="regex",
            compact=compact,
            figsize=figsize,
        )

    @property
    def site_names(self) -> list[Any]:
        return list(self.posterior_samples_.keys())


def init_params(distributions) -> dict:
    """
    Initializes the model parameters.
    """

    params = {}

    for coefficient_name, distribution in distributions.items():

        if not isinstance(distribution, OrderedDict):
            params[coefficient_name] = numpyro.sample(coefficient_name, distribution)
        else:
            coefficients = []
            for (
                inner_coefficient_name,
                coefficient_distribution,
            ) in distribution.items():
                coefficients.append(
                    numpyro.sample(inner_coefficient_name, coefficient_distribution)
                )
            params[coefficient_name] = jnp.concatenate(coefficients, axis=0)

    return params


class ExogenousEffectMixin:

    def __init__(self,
                 default_effect_mode,
                 exogenous_effects: Dict[str, Tuple[str, Any]],
                 default_exogenous_prior: Tuple[str, Any],
                 **kwargs):

        self.default_effect_mode = default_effect_mode
        self.exogenous_effects = exogenous_effects
        self.default_exogenous_prior = default_exogenous_prior
        super().__init__(**kwargs)

    def _set_custom_effects(self, feature_names):

        effects_and_columns = {}
        columns_with_effects = set()
        exogenous_effects = self.exogenous_effects or {}

        for effect_name, (column_regex, effect) in exogenous_effects.items():

            columns = [
                column for column in feature_names if re.match(column_regex, column)
            ]

            if columns_with_effects.intersection(columns):
                raise ValueError(
                    "Columns {} are already set".format(
                        columns_with_effects.intersection(columns)
                    )
                )

            if not len(columns):
                logging.warning("No columns match the regex {}".format(column_regex))

            columns_with_effects = columns_with_effects.union(columns)

            effects_and_columns.update(
                {
                    effect_name: (
                        columns,
                        effect,
                    )
                }
            )

        features_without_effects: set = feature_names.difference(columns_with_effects)

        if len(features_without_effects):

            default_dist = getattr(dist, self.default_exogenous_prior[0])
            args = self.default_exogenous_prior[1:]
            effects_and_columns.update(
                {
                    "default": (
                        features_without_effects,
                        LinearEffect(
                            id="exog",
                            prior=(default_dist, *args),
                            effect_mode=self.default_effect_mode,
                        ),
                    )
                }
            )

        self._exogenous_effects_and_columns = effects_and_columns

    def _get_exogenous_data_array(self, X):

        out = {}
        for effect_name, (columns, _) in self._exogenous_effects_and_columns.items():
            # If no columns are found, skip
            if len(columns) == 0:
                continue
            
            if X.index.nlevels == 1:
                array = jnp.array(X[columns].values)
            else:
                array = series_to_tensor(X[columns])

            out[effect_name] = array 

        return out

    @property
    def exogenous_effect_dict(self):
        return {k: v[1] for k, v in self._exogenous_effects_and_columns.items() if len(v[0])}
