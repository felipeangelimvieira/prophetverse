"""Contains the implementation of the HierarchicalProphet forecaster."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

from prophetverse.effects.target.multivariate import MultivariateNormal
from prophetverse.sktime.base import BaseEffect, BaseProphetForecaster
from prophetverse.utils import loc_bottom_series, reindex_time_series, series_to_tensor

from ._expand_column_per_level import ExpandColumnPerLevel


class HierarchicalProphet(BaseProphetForecaster):
    """A Bayesian hierarchical time series forecasting model based on Meta's Prophet.

    This method forecasts all bottom series in a hierarchy at once, using a
    MultivariateNormal as the likelihood function and LKJ priors for the correlation
    matrix.

    This forecaster is particularly interesting if you want to fit shared coefficients
    across series. In that case, `shared_features` parameter should be a list of
    feature names that should have that behaviour.

    Parameters
    ----------
    trend : Union[BaseEffect, str], default="linear"
        Trend component of the model.
    feature_transformer : BaseTransformer, default=None
        Transformer for features preprocessing.
    exogenous_effects : optional, default=None
        Effects to model exogenous variables.
    default_effect : optional, default=None
        Default effect specification.
    shared_features : optional, default=None
        Features shared across time series.
    noise_scale : float, default=0.05
        Scale parameter for the noise distribution.
    correlation_matrix_concentration : float, default=1.0
        Concentration parameter for the correlation matrix.
    rng_key : optional, default=None
        Random number generator key.
    inference_engine : optional, default=None
        Engine used for inference.

    Examples
    --------
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> from prophetverse.sktime.multivariate import HierarchicalProphet
    >>> agg = Aggregator()
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=3,
    ...     no_levels=1,
    ...     random_seed=123,
    ...     length=7,
    ... )
    >>> y = agg.fit_transform(y)
    >>> forecaster = HierarchicalProphet()
    >>> forecaster = forecaster.fit(y)
    >>> y_pred = forecaster.predict(fh=[1])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse",
        # estimator type
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "handles-missing-data": False,
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
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "fit_is_empty": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        trend: Union[BaseEffect, str] = "linear",
        feature_transformer: BaseTransformer = None,
        exogenous_effects=None,
        default_effect=None,
        shared_features=None,
        noise_scale=0.05,
        correlation_matrix_concentration=1.0,
        rng_key=None,
        inference_engine=None,
        likelihood=None,
    ):

        self.noise_scale = noise_scale
        self.shared_features = shared_features
        self.feature_transformer = feature_transformer
        self.correlation_matrix_concentration = correlation_matrix_concentration
        self.likelihood = likelihood

        super().__init__(
            # Trend
            trend=trend,
            # Exog effects
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
            # Base Bayesian forecaster
            rng_key=rng_key,
            inference_engine=inference_engine,
        )

        self._validate_hyperparams()

    def _validate_hyperparams(self):
        """Validate the hyperparameters of the HierarchicalProphet forecaster."""
        super()._validate_hyperparams()

        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be greater than 0.")
        if self.correlation_matrix_concentration <= 0:
            raise ValueError("correlation_matrix_concentration must be greater than 0.")

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
        self.trend_model_ = self._trend.clone()
        self.trend_model_.fit(X=X_bottom, y=y_bottom, scale=self._scale)

        if self.likelihood is not None:
            self.likelihood_model_ = self.likelihood.clone()
        else:
            self.likelihood_model_ = MultivariateNormal(
                noise_scale=self.noise_scale,
                correlation_matrix_concentration=self.correlation_matrix_concentration,
            )

        self.likelihood_model_.fit(X=X_bottom, y=y_bottom, scale=self._scale)

        trend_data = self.trend_model_.transform(X=X_bottom, fh=fh)
        target_data = self.likelihood_model_.transform(X=y_bottom, fh=fh)

        self._fit_effects(X_bottom, y_bottom)
        exogenous_data = self._transform_effects(X_bottom, fh=fh)

        self.fit_and_predict_data_ = {
            "trend_model": self.trend_model_,
            "target_model": self.likelihood_model_,
            "exogenous_effects": self.non_skipped_exogenous_effect,
        }

        return dict(
            y=y_bottom_arrays,
            data=exogenous_data,
            trend_data=trend_data,
            target_data=target_data,
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
        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        if X is None or X.shape[1] == 0:
            idx = reindex_time_series(self._y, fh_as_index).index
            X = pd.DataFrame(index=idx)
            X = self.aggregator_.transform(X)

        X_bottom = loc_bottom_series(X)

        if self._has_exogenous_variables:

            assert fh_as_index.isin(
                X_bottom.index.get_level_values(-1)
            ).all(), "Missing exogenous variables for some series or dates."
            if self.feature_transformer is not None:
                X_bottom = self.feature_transformer.transform(X_bottom)
            X_bottom = self.expand_columns_transformer_.transform(X_bottom)

        trend_data = self.trend_model_.transform(X=X_bottom, fh=fh_as_index)
        target_data = self.likelihood_model_.transform(X=None, fh=fh_as_index)

        exogenous_data = self._transform_effects(X=X_bottom, fh=fh_as_index)

        return dict(
            y=None,
            data=exogenous_data,
            trend_data=trend_data,
            target_data=target_data,
            **self.fit_and_predict_data_,
        )

    def predict_samples(
        self, fh: ForecastingHorizon, X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generate samples for the given exogenous variables and forecasting horizon.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

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

    def _postprocess_output(self, y: pd.DataFrame) -> pd.DataFrame:
        """Postprocess outputs, by aggregating them.

        Parameters
        ----------
        y : pd.DataFrame
            dataframe with output predictions

        Returns
        -------
        pd.DataFrame
            postprocessed dataframe

        """
        return self.aggregator_.transform(y)

    @classmethod
    def get_test_params(cls, parameter_set="default") -> List[dict[str, Any]]:
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
        from prophetverse.engine.prior import PriorPredictiveInferenceEngine
        from prophetverse.engine.mcmc import MCMCInferenceEngine
        from prophetverse.engine.optimizer import AdamOptimizer

        return [
            {"inference_engine": PriorPredictiveInferenceEngine(1)},
            {
                "inference_engine": MCMCInferenceEngine(
                    num_samples=1, num_warmup=1, num_chains=1
                ),
            },
        ]
