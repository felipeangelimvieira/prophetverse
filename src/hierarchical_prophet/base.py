import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

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
from hierarchical_prophet.exogenous_priors import get_exogenous_priors
import arviz as az


class BaseBayesianForecaster(BaseForecaster):
    """
    Base class for Bayesian forecasters in hierarchical-prophet.

    Args:
        rng_seed (int): Random number generator seed.
        method (str): Inference method to use. Currently, only "mcmc" is supported.
        num_samples (int): Number of MCMC samples to draw.
        num_warmup (int): Number of warmup steps for MCMC.
        num_chains (int): Number of MCMC chains to run.
        prefix (str): Prefix to add to sample site names.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    _tags = {
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        rng_key=1000,
        method="mcmc",
        mcmc_samples=2000,
        mcmc_warmup=100,
        mcmc_chains=1,
        prefix="",
        *args,
        **kwargs,
    ):
        """
        Initialize the BaseBayesianForecaster.

        Args:
            rng_seed (int): Random number generator seed.
            method (str): Inference method to use. Currently, only "mcmc" is supported.
            num_samples (int): Number of MCMC samples to draw.
            num_warmup (int): Number of warmup steps for MCMC.
            num_chains (int): Number of MCMC chains to run.
            prefix (str): Prefix to add to sample site names.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.rng_key = rng_key
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.mcmc_chains = mcmc_chains
        self.method = method
        self.prefix = prefix
        self._sample_sites = set()
        super().__init__(*args, **kwargs)
        self.predictive_samples_ = None

    def _get_numpyro_model_data(self, y, X, fh) -> Dict[str, Any]:
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

    def _predict_samples(self, X, fh):
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

        if self.method == "mcmc":
            self._fit_mcmc(y, X, fh)
        else:
            raise ValueError(f"Unknown method {self.method}")

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
        predictive_samples = self.predict_samples(X, fh)
        self.forecast_samples_ = predictive_samples
        y_pred = predictive_samples.mean(axis=1).to_frame(self._y.columns[0])

        return y_pred

    def predict_samples(self, X, fh):
        """
        Generate samples from the posterior predictive distribution.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            pd.DataFrame: Samples from the posterior predictive distribution.
        """
        fh_as_index = self.fh_to_index(fh)
        self.predictive_samples_ = self._predict_samples(X, fh)

        n_samples = self.predictive_samples_["obs"].shape[0]
        return pd.DataFrame(
            data=self.predictive_samples_["obs"].T.reshape((-1, n_samples)),
            columns=list(range(n_samples)),
            index=self.periodindex_to_multiindex(fh_as_index),
        )

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

        forecast_samples_ = self.predict_samples(X, fh)

        var_names = self._get_varnames()
        var_name = var_names[0]
        int_idx = pd.MultiIndex.from_product([var_names, alpha])

        quantiles = forecast_samples_.quantile(alpha, axis=1).T
        quantiles.columns = int_idx

        return quantiles

    def _fit_mcmc(self, y, X, fh):
        """
        Fit the Bayesian forecaster using MCMC.

        Args:
            y (pd.DataFrame): Target variable.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.
        """
        data = self._get_numpyro_model_data(y, X, fh)

        self.distributions_ = data.get("distributions", {})

        nuts_kernel = NUTS(self.model, dense_mass=True, init_strategy=init_to_mean())
        self.mcmc_ = MCMC(
            nuts_kernel,
            num_samples=self.mcmc_samples,
            num_warmup=self.mcmc_warmup,
            num_chains=self.mcmc_chains,
        )

        self.mcmc_.run(
            self.rng_key,
            **data,
        )

        self.posterior_samples_ = self.mcmc_.get_samples()

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
            az.from_numpyro(self.mcmc_),
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
