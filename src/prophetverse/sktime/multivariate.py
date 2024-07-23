"""Contains the implementation of the HierarchicalProphet forecaster."""

from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

from prophetverse.effects.trend import (
    FlatTrend,
    PiecewiseLinearTrend,
    PiecewiseLogisticTrend,
)
from prophetverse.models import multivariate_model
from prophetverse.sktime.base import BaseEffect, BaseEffectsBayesianForecaster
from prophetverse.utils import loc_bottom_series, reindex_time_series, series_to_tensor

from ._expand_column_per_level import ExpandColumnPerLevel


class HierarchicalProphet(BaseEffectsBayesianForecaster):
    """A Bayesian hierarchical time series forecasting model based on the Prophet.

    This class forecasts all series in a hierarchy at once, using a MultivariateNormal
    as the likelihood function and LKJ priors for the correlation matrix.

    This class may be interesting if you want to fit shared coefficients across series.
    By default, all coefficients are obtained exclusively for each series, but this can
    be changed through the `shared_coefficients` parameter.

    Parameters
    ----------
    changepoint_interval : int
        The number of points between each potential changepoint.
    changepoint_range : float
        Proportion of the history in which trend changepoints will be estimated.
        If a float between 0 and 1, the range will be that proportion of the history.
        If an int, the range will be that number of points. A negative int indicates the
        number of points counting from the end of the history.
    changepoint_prior_scale : float
        Parameter controlling the flexibility of the automatic changepoint selection.
    offset_prior_scale : float, optional, default=0.1
        Scale parameter for the prior distribution of the offset.
    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the capacity prior.
    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the capacity prior.
    trend : str, optional, default='linear'
        Type of trend. Either "linear" or "logistic".
    feature_transformer : BaseTransformer or None, optional, default=None
        A transformer to preprocess the exogenous features.
    exogenous_effects : list of AbstractEffect, optional, default=None
        A list defining the exogenous effects to be used in the model.
    default_effect : AbstractEffect, optional, default=None
        The default effect to be used when no effect is specified for a variable.
    shared_features : list, optional, default=[]
        List of shared features across series.
    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.
    mcmc_warmup : int, optional, default=200
        Number of warmup steps for MCMC.
    mcmc_chains : int, optional, default=4
        Number of MCMC chains.
    inference_method : str, optional, default='map'
        Inference method to use. Either "map" or "mcmc".
    optimizer_name : str, optional, default='Adam'
        Name of the optimizer to use.
    optimizer_kwargs : dict, optional, default={'step_size': 1e-4}
        Additional keyword arguments for the optimizer.
    optimizer_steps : int, optional, default=100_000
        Number of optimization steps.
    noise_scale : float, optional, default=0.05
        Scale parameter for the noise.
    correlation_matrix_concentration : float, optional, default=1.0
        Concentration parameter for the correlation matrix.
    rng_key : jax.random.PRNGKey, optional, default=None
        Random number generator key.
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
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

        self.model = multivariate_model  # type: ignore[method-assign]
        self._validate_hyperparams()

    def _validate_hyperparams(self):
        """Validate the hyperparameters of the HierarchicalProphet forecaster."""
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
            raise ValueError("correlation_matrix_concentration must be greater than 0.")

        if self.trend not in ["linear", "logistic"]:
            raise ValueError('trend must be either "linear" or "logistic".')

    def _get_fit_data(self, y, X, fh):
        """
        Prepare the data for the NumPyro model.

        Parameters
        ----------
        y: pd.DataFrame
            Training target time series.
        X: pd.DataFrame
            Training exogenous variables.
        fh: ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        dict
            A dictionary containing the model data.
        """
        # Handling series without __total indexes
        self.aggregator_ = Aggregator()
        self.original_y_indexes_ = y.index
        fh = y.index.get_level_values(-1).unique()
        y = self.aggregator_.fit_transform(y)

        # Updating internal _y of sktime because BaseBayesianForecaster
        # uses it to convert
        # Forecast Horizon into multiindex correcly
        self.internal_y_indexes_ = y.index

        # Convert inputs to array, including the time index
        y_bottom = loc_bottom_series(y)
        y_bottom_arrays = series_to_tensor(y_bottom)

        # If no exogenous variables, create empty DataFrame
        # Else, aggregate exogenous variables and transform them
        if X is None or X.columns.empty:
            X = pd.DataFrame(index=y.index)

        X_bottom = loc_bottom_series(X)

        if self.feature_transformer is not None:
            X_bottom = self.feature_transformer.fit_transform(X_bottom)

        self._has_exogenous_variables = (
            X_bottom is not None and not X_bottom.columns.empty
        )

        if self._has_exogenous_variables:
            shared_features = self.shared_features
            if shared_features is None:
                shared_features = []

            self.expand_columns_transformer_ = ExpandColumnPerLevel(
                X_bottom.columns.difference(shared_features).to_list()
            ).fit(X_bottom)
            X_bottom = self.expand_columns_transformer_.transform(X_bottom)

        else:
            self._exogenous_effects_and_columns = {}
            exogenous_data = {}

        # Trend model
        self.trend_model_ = self._get_trend_model()
        self.trend_model_.fit(X=X_bottom, y=y_bottom, scale=self._scale)
        trend_data = self.trend_model_.transform(X=X_bottom, fh=fh)

        self._fit_effects(X_bottom, y_bottom)
        exogenous_data = self._transform_effects(X_bottom, fh=fh)

        self.fit_and_predict_data_ = {
            "trend_model": self.trend_model_,
            "exogenous_effects": self.non_skipped_exogenous_effect,
            "correlation_matrix_concentration": self.correlation_matrix_concentration,
            "noise_scale": self.noise_scale,
            "is_single_series": self.n_series == 1,
        }

        return dict(
            y=y_bottom_arrays,
            data=exogenous_data,
            trend_data=trend_data,
            **self.fit_and_predict_data_,
        )

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Parameters
        ----------
        X: pd.DataFrame
            Exogenous variables.
        fh: ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        np.ndarray
            Predicted samples.
        """
        fh_dates = fh.to_absolute(
            cutoff=self.internal_y_indexes_.get_level_values(-1).max()
        )
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        if X is None or X.shape[1] == 0:
            idx = reindex_time_series(self._y, fh_as_index).index
            X = pd.DataFrame(index=idx)
            X = self.aggregator_.transform(X)

        if self._has_exogenous_variables:

            assert fh_as_index.isin(
                X.index.get_level_values(-1)
            ).all(), "Missing exogenous variables for some series or dates."
            if self.feature_transformer is not None:
                X = self.feature_transformer.transform(X)
            X = self.expand_columns_transformer_.transform(X)

        trend_data = self.trend_model_.transform(X=loc_bottom_series(X), fh=fh_as_index)
        exogenous_data = self._transform_effects(X=loc_bottom_series(X), fh=fh_as_index)

        return dict(
            y=None,
            data=exogenous_data,
            trend_data=trend_data,
            **self.fit_and_predict_data_,
        )

    def _get_trend_model(self):
        """
        Return the trend model based on the specified trend parameter.

        Returns
        -------
        BaseEffect
            The trend model based on the specified trend parameter.

        Raises
        ------
        ValueError
            If the trend parameter is not one of 'linear', 'logistic', 'flat'
            or a BaseEffect instance.
        """
        # Changepoints and trend
        if self.trend == "linear":
            return PiecewiseLinearTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
            )

        elif self.trend == "logistic":
            return PiecewiseLogisticTrend(
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
            )
        elif self.trend == "flat":
            return FlatTrend(changepoint_prior_scale=self.changepoint_prior_scale)

        elif isinstance(self.trend, BaseEffect):
            return self.trend

        raise ValueError(
            "trend must be either 'linear', 'logistic' or a BaseEffect instance."
        )

    def _get_exogenous_matrix_from_X(self, X: pd.DataFrame) -> jnp.ndarray:
        """
        Convert the exogenous variables to a NumPyro matrix.

        Parameters
        ----------
        X: pd.DataFrame
            The exogenous variables.

        Return
        ------
        jnp.ndarray
            The NumPyro matrix of the exogenous variables.
        """
        X_bottom = loc_bottom_series(X)
        X_arrays = series_to_tensor(X_bottom)

        return X_arrays

    def predict_samples(
        self, fh: ForecastingHorizon, X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Parameters
        ----------
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns
        -------
        np.ndarray
            Predicted samples.
        """
        samples = super().predict_samples(X=X, fh=fh)

        return self.aggregator_.transform(samples)

    def _filter_series_tuples(self, levels: List[Tuple]) -> List[Tuple]:
        """Filter series tuples, returning only series of interest.

        Since this class performs a bottom-up aggregation, we are only interested in the
        bottom levels of the hierarchy. This method filters the series tuples, returning
        only the bottom levels.

        Parameters
        ----------
        levels : List[Tuple]
            The original levels of timeseries (`y.index.droplevel(-1).unique()`)

        Returns
        -------
        List[Tuple]
            The same object as `levels`, but with only the bottom levels.
        """
        # Make it a tuple for consistency
        if not isinstance(levels[0], (tuple, list)):
            levels = [(idx,) for idx in levels]

        bottom_levels = [idx for idx in levels if idx[-1] != "__total"]
        return bottom_levels

    @property
    def n_series(self):
        """Get the number of series.

        Returns
        -------
        int
            Number of series.
        """
        if self.internal_y_indexes_.nlevels == 1:
            return 1
        return len(
            self._filter_series_tuples(
                self.internal_y_indexes_.droplevel(-1).unique().tolist()
            )
        )

    @classmethod
    def get_test_params(cls, parameter_set="default") -> List[dict[str, int]]:
        """Params to be used in sktime unit tests.

        Parameters
        ----------
        parameter_set : str, optional
            The parameter set to be used (ignored in this implementation)

        Returns
        -------
        List[dict[str, int]]
            A list of dictionaries containing the test parameters.
        """
        return [
            {
                "optimizer_steps": 1_000,
            }
        ]
