import functools
import itertools
import logging
import re
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import lax, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import _get_s_matrix

from prophetverse.models import multivariate_model
from prophetverse.sktime.base import (BaseBayesianForecaster,
                                      ExogenousEffectMixin)
from prophetverse.trend.piecewise import (PiecewiseLinearTrend,
                                          PiecewiseLogisticTrend, TrendModel)
from prophetverse.utils.frame_to_array import (loc_bottom_series,
                                               series_to_tensor)
from prophetverse.utils.multiindex import reindex_time_series

from ._expand_column_per_level import ExpandColumnPerLevel

logger = logging.getLogger("sktime-numpyro")


class HierarchicalProphet(ExogenousEffectMixin, BaseBayesianForecaster):
    """A class that represents a Bayesian hierarchical time series forecasting model based on the Prophet algorithm.

    This class forecasts all series in a hierarchy at once, using a MultivariateNormal as the likelihood function, and
    LKJ priors for the correlation matrix.

    This class may be interesting if you want to fit shared coefficients across series. By default, all coefficients are
    obtained exclusively for each series, but this can be changed through the `shared_coefficients` parameter.

    Args:
        changepoint_interval (int): The number of points between each potential changepoint.
        changepoint_range (float): Proportion of the history in which trend changepoints will be estimated.
                                   If a float between 0 and 1, the range will be that proportion of the history.
                                   If an int, the range will be that number of points. A negative int indicates number of points
                                   counting from the end of the history.
        changepoint_prior_scale (float): Parameter controlling the flexibility of the automatic changepoint selection.
        offset_prior_scale (float): Scale parameter for the prior distribution of the offset. Default is 0.1.
        capacity_prior_scale (float): Scale parameter for the capacity prior. Defaults to 0.2.
        capacity_prior_loc (float): Location parameter for the capacity prior. Defaults to 1.1.
        trend (str): Type of trend. Either "linear" or "logistic". Defaults to "linear".
        feature_transformer (BaseTransformer or None): A transformer to preprocess the exogenous features. Defaults to None.
        exogenous_effects (List[AbstractEffect]): A list defining the exogenous effects to be used in the model.
        default_effect (AbstractEffect): The default effect to be used when no effect is specified for a variable.
        shared_features (list): List of shared features across series. Defaults to an empty list.
        mcmc_samples (int): Number of MCMC samples to draw. Defaults to 2000.
        mcmc_warmup (int): Number of warmup steps for MCMC. Defaults to 200.
        mcmc_chains (int): Number of MCMC chains. Defaults to 4.
        inference_method (str): Inference method to use. Either "map" or "mcmc". Defaults to "map".
        optimizer_name (str): Name of the optimizer to use. Defaults to "Adam".
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer. Defaults to {"step_size": 1e-4}.
        optimizer_steps (int): Number of optimization steps. Defaults to 100_000.
        noise_scale (float): Scale parameter for the noise. Defaults to 0.05.
        correlation_matrix_concentration (float): Concentration parameter for the correlation matrix. Defaults to 1.0.
        rng_key (jax.random.PRNGKey): Random number generator key. Defaults to random.PRNGKey(24).
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            'pd.DataFrame',
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            'pd.DataFrame',
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "fit_is_empty": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.1,
        offset_prior_scale=0.1,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        trend="linear",
        feature_transformer: BaseTransformer = None,
        exogenous_effects=None,
        default_effect=None,
        shared_features=None,
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        noise_scale=0.05,
        correlation_matrix_concentration=1.0,
        rng_key=None,
    ):

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.noise_scale = noise_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend
        self.shared_features = shared_features
        self.feature_transformer = feature_transformer
        self.correlation_matrix_concentration = correlation_matrix_concentration

        super().__init__(
            rng_key=rng_key,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
        )

        self.model = multivariate_model
        self._validate_hyperparams()

    def _validate_hyperparams(self):
        """
        Validate the hyperparameters of the HierarchicalProphet forecaster.
        """
        if self.changepoint_interval <= 0:
            raise ValueError("changepoint_interval must be greater than 0.")

        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be greater than 0.")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")
        if self.capacity_prior_scale <= 0:
            raise ValueError("capacity_prior_scale must be greater than 0.")
        if self.capacity_prior_loc <= 0:
            raise ValueError("capacity_prior_loc must be greater than 0.")
        if self.offset_prior_scale <= 0:
            raise ValueError("offset_prior_scale must be greater than 0.")
        if self.correlation_matrix_concentration <= 0:
            raise ValueError(
                "correlation_matrix_concentration must be greater than 0."
            )

        if self.trend not in ["linear", "logistic"]:
            raise ValueError('trend must be either "linear" or "logistic".')

    def _get_fit_data(self, y, X, fh):
        """
        Prepare the data for the NumPyro model.

        Args:
            y (pd.DataFrame): Training target time series.
            X (pd.DataFrame): Training exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: A dictionary containing the model data.
        """

        # Handling series without __total indexes
        self.aggregator_ = Aggregator()
        self.original_y_indexes_ = y.index
        y = self.aggregator_.fit_transform(y)

        # Updating internal _y of sktime because BaseBayesianForecaster uses it to convert
        # Forecast Horizon into multiindex correcly
        self.internal_y_indexes_ = y.index

        # Convert inputs to array, including the time index
        y_bottom = loc_bottom_series(y)
        y_bottom_arrays = series_to_tensor(y_bottom)

        ## Changepoints and trend
        if self.trend == "linear":
            self.trend_model_ = PiecewiseLinearTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
                squeeze_if_single_series=False,
            )

        elif self.trend == "logistic":
            self.trend_model_ = PiecewiseLogisticTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
                capacity_prior=dist.TransformedDistribution(
                    dist.HalfNormal(self.capacity_prior_scale),
                    dist.transforms.AffineTransform(
                        loc=self.capacity_prior_loc, scale=1
                    ),
                ),
                squeeze_if_single_series=False,
            )

        elif isinstance(self.trend, TrendModel):
            self.trend_model_ = self.trend
        else:
            raise ValueError("trend must be either 'linear', 'logistic' or a TrendModel instance.")

        self.trend_model_.initialize(y_bottom)
        fh = y.index.get_level_values(-1).unique()
        trend_data = self.trend_model_.prepare_input_data(fh)

        # Exog variables

        # If no exogenous variables, create empty DataFrame
        # Else, aggregate exogenous variables and transform them
        if (X is None or X.columns.empty) and self.feature_transformer is not None:
            X = pd.DataFrame(index=y.index)
        if self.feature_transformer is not None:
            X = self.feature_transformer.fit_transform(X)
        self._has_exogenous_variables = X is not None and not X.columns.empty

        if self._has_exogenous_variables:
            shared_features = self.shared_features
            if shared_features is None:
                shared_features = []

            self.expand_columns_transformer_ = ExpandColumnPerLevel(
                X.columns.difference(shared_features).to_list()
            ).fit(X)
            X = X.loc[y_bottom.index]
            X = self.expand_columns_transformer_.transform(X)

            self._set_custom_effects(feature_names=X.columns)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))

        else:
            self._exogenous_effects_and_columns = {}
            exogenous_data = {}

        self.fit_and_predict_data_ = {
            "noise_scale" : self.noise_scale,
            "trend_model" : self.trend_model_,
            "exogenous_effects": self.exogenous_effect_dict,
            "correlation_matrix_concentration": self.correlation_matrix_concentration,
            "noise_scale": self.noise_scale,
            "is_single_series" : self.n_series == 1
        }

        return dict(
            y=y_bottom_arrays,
            data=exogenous_data,
            trend_data = trend_data,
            
            **self.fit_and_predict_data_,
        )

    def predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """
        samples = super().predict_samples(X=X, fh=fh)

        return self.aggregator_.transform(samples)

    def _get_exogenous_matrix_from_X(self, X: pd.DataFrame) -> jnp.ndarray:
        """
        Convert the exogenous variables to a NumPyro matrix.

        Args:
            X (pd.DataFrame): The exogenous variables.

        Returns:
            jnp.ndarray: The NumPyro matrix of the exogenous variables.
        """
        X_bottom = loc_bottom_series(X)
        X_arrays = series_to_tensor(X_bottom)

        return X_arrays

    def predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """
        samples = super().predict_samples(X=X, fh=fh)

        return self.aggregator_.transform(samples)

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            np.ndarray: Predicted samples.
        """

        fh_dates = fh.to_absolute(
            cutoff=self.internal_y_indexes_.get_level_values(-1).max()
        )
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        trend_data = self.trend_model_.prepare_input_data(fh_as_index)

        if self._has_exogenous_variables:
            if X is None or X.shape[1] == 0:
                idx = reindex_time_series(self._y, fh_as_index).index
                X = pd.DataFrame(index=idx)
                X = self.aggregator_.transform(X)

            X = X.loc[X.index.get_level_values(-1).isin(fh_as_index)]

            assert X.index.get_level_values(-1).nunique() == fh_as_index.nunique(), "Missing exogenous variables for some series or dates."
            if self.feature_transformer is not None:
                X = self.feature_transformer.transform(X)
            X = self.expand_columns_transformer_.transform(X)
            exogenous_data = self._get_exogenous_data_array(loc_bottom_series(X))
        else:
            exogenous_data = {}

        return dict(
            y=None,
            data=exogenous_data,
            trend_data=trend_data,
            **self.fit_and_predict_data_,
        )

    def _filter_series_tuples(self, levels: List[Tuple]) -> List[Tuple]:

        # Make it a tuple for consistency
        if not isinstance(levels[0], (tuple, list)):
            levels = [(idx,) for idx in levels]

        bottom_levels = [idx for idx in levels if idx[-1] != "__total"]
        return bottom_levels

    @property
    def n_series(self):
        """Get the number of series.

        Returns:
            int: Number of series.
        """
        if self.internal_y_indexes_.nlevels == 1:
            return 1
        return len(self._filter_series_tuples(self.internal_y_indexes_.droplevel(-1).unique().tolist()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        
        return [{
            "optimizer_steps":1_000,
        }]
