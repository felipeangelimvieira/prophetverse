"""Base classes for sktime forecasters in prophetverse."""

import itertools
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from prophetverse.effects.base import BaseEffect
from prophetverse.effects.linear import LinearEffect
from prophetverse.effects.trend import (
    FlatTrend,
    PiecewiseLinearTrend,
    PiecewiseLogisticTrend,
)
from prophetverse.engine import (
    BaseInferenceEngine,
    MAPInferenceEngine,
    MCMCInferenceEngine,
)
from prophetverse.engine.optimizer.optimizer import (
    CosineScheduleAdamOptimizer,
    _LegacyNumpyroOptimizer,
)
from prophetverse.utils import get_multiindex_loc
from prophetverse.utils.deprecation import deprecation_warning


class BaseBayesianForecaster(BaseForecaster):
    """

    A base class for numpyro Bayesian forecasters in sktime.

    Specifies methods and signatures that all Bayesian forecasters have to implement,
    and handles optimization and sampling of the posterior distribution.


    Parameters
    ----------
    rng_key: KeyArray
        The RNG Key to use for sampling.
    inference_method: str
        The inference method to use. Can be either "mcmc" or "map".
    mcmc_samples: int
        The number of MCMC samples to draw.
    mcmc_warmup: int
        The number of warmup steps for MCMC.
    mcmc_chains: int
        The number of MCMC chains to run.
    optimizer_steps: int
        The number of optimization steps to run, in case of MAP inference.
    optimizer_name: str
        The name of the optimizer to use, in case of MAP inference. Should be the name
        of a optimizer in the `numpyro.optim` module.
        "optax" uses a cosine decay schedule.
    optimizer_kwargs: dict
        Additional keyword arguments to pass to the optimizer.
    scale: float or pd.Series, optional
        The scale of the target variable. If not provided, it will be inferred from the
        training data.


    """

    _tags = {
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "enforce_index_type": [pd.Period, pd.DatetimeIndex],
    }

    def __init__(
        self,
        rng_key: jax.typing.ArrayLike = None,
        inference_method: str = "map",
        mcmc_samples: int = 2000,
        mcmc_warmup: int = 200,
        mcmc_chains: int = 4,
        optimizer_steps: int = 100_000,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        scale=None,
        inference_engine: Optional[BaseInferenceEngine] = None,
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
        self.scale = scale
        self.inference_engine = inference_engine
        super().__init__()

        if self.scale:
            self._scale = self.scale
        else:
            self._scale = None

        if self.inference_engine is not None:
            self._inference_engine = self.inference_engine
        else:
            self._inference_engine = self._get_inference_engine()

    def _get_inference_engine(self):
        """Temporarily return the inference engine.

        Returns
        -------
        BaseInferenceEngine
            The inference engine.
        """
        deprecation_warning(
            "inference_method",
            "0.5.0",
            "Use the `inference_engine` parameter instead.",
        )

        if self.inference_method == "map":
            optimizer = self._optimizer()

            return MAPInferenceEngine(
                optimizer=optimizer,
                num_steps=self.optimizer_steps,
                rng_key=self.rng_key,
            )
        elif self.inference_method == "mcmc":
            return MCMCInferenceEngine(
                num_samples=self.mcmc_samples,
                num_warmup=self.mcmc_warmup,
                num_chains=self.mcmc_chains,
                rng_key=self.rng_key,
                dense_mass=False,
            )
        else:
            raise ValueError(f"Unknown method {self.inference_method}")

    @property
    def _likelihood_is_discrete(self):
        """Property that indicates whether the forecaster uses a discrete likelihood.

        As a consequence, the target variable must be integer-valued and will not be
        scaled before _get_fit_data is called.

        Returns
        -------
        bool
            True if the forecaster uses a discrete likelihood, False otherwise.
        """
        return False

    def _optimizer(self) -> numpyro.optim._NumPyroOptim:
        """Return the optimizer.

        Returns
        -------
        _NumPyroOptim
            An instance of the optimizer.
        """
        optimizer_kwargs = self.optimizer_kwargs
        optimizer_name = self.optimizer_name
        rng_key = self.rng_key

        if rng_key is None:
            rng_key = jax.random.PRNGKey(24)

        if optimizer_kwargs is None:
            optimizer_kwargs = {"step_size": 1e-4}
        if optimizer_name is None:
            optimizer_name = "Adam"

        if optimizer_name.startswith("optax"):

            return CosineScheduleAdamOptimizer(**optimizer_kwargs)

        return _LegacyNumpyroOptimizer(
            optimizer_name=optimizer_name, optimizer_kwargs=optimizer_kwargs
        )

    # pragma: no cover
    def _get_fit_data(
        self, y: pd.DataFrame, X: pd.DataFrame, fh: ForecastingHorizon
    ) -> Dict[str, Any]:
        """
        Get the data required for the Numpyro model.

        Parameters
        ----------
        y : pd.DataFrame
            Target variable.
        X : pd.DataFrame
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        Dict[str, Any]
            Data required for the Numpyro model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    # pragma: no cover
    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon):
        """Generate samples from the posterior predictive distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        pd.DataFrame
            Samples from the posterior predictive distribution.
        """
        raise NotImplementedError("Must be implemented by subclass")

    # pragma: no cover
    def model(self, *args, **kwargs):
        """
        Numpyro model.

        This method must be implemented by the subclass, or overriden as
        a class attribute.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        Any
            Model output.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def _fit(self, y, X, fh):
        """
        Fit the Bayesian forecaster to the training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target variable.
        X : pd.DataFrame
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        self : object
            The fitted Bayesian forecaster.
        """
        self._set_y_scales(y)
        y = self._scale_y(y)

        self.internal_y_indexes_ = y.index
        data = self._get_fit_data(y, X, fh)

        self.distributions_ = data.get("distributions", {})

        rng_key = self.rng_key
        if rng_key is None:
            rng_key = jax.random.PRNGKey(24)

        self.inference_engine_ = self._inference_engine.clone()
        self.inference_engine_.infer(self.model, **data)
        self.posterior_samples_ = self.inference_engine_.posterior_samples_

        return self

    def _predict(self, fh, X):
        """
        Generate point forecasts for the given forecasting horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame
            Exogenous variables.

        Returns
        -------
        pd.DataFrame
            Point forecasts for the forecasting horizon.
        """
        predictive_samples = self.predict_components(fh=fh, X=X)
        mean = predictive_samples["mean"]
        y_pred = mean.to_frame(self._y.columns[0])

        return self._postprocess_output(y_pred)

    def predict_all_sites(self, fh: ForecastingHorizon, X: pd.DataFrame = None):
        """
        Predicts the values for all sites.

        Given a forecast horizon and optional input features, returns a DataFrame
        the mean of the predicted values for all sites.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon, specifying the number of time steps to predict into
            the future.
        X : array-like, optional
            The input features used for prediction. Defaults to None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the predicted values for all sites, with the site
            names as columns and the forecast horizon as the index.
        """
        deprecation_warning(
            "predict_all_sites",
            "0.5.0",
            "Use the `predict_components` method instead.",
        )
        return self.predict_components(fh=fh, X=X)

    def predict_components(self, fh: ForecastingHorizon, X: pd.DataFrame = None):
        """
        Predicts the values for all sites.

        Given a forecast horizon and optional input features, returns a DataFrame
        the mean of the predicted values for all sites.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon, specifying the number of time steps to predict into
            the future.
        X : array-like, optional
            The input features used for prediction. Defaults to None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the predicted values for all sites, with the site
            names as columns and the forecast horizon as the index.
        """
        if self._is_vectorized:
            return self._vectorize_predict_method("predict_all_sites", X=X, fh=fh)

        fh_as_index = self.fh_to_index(fh)
        predictive_samples_ = self._get_predictive_samples_dict(fh=fh, X=X)

        out = pd.DataFrame(
            data={
                site: data.mean(axis=0).flatten()
                for site, data in predictive_samples_.items()
            },
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()

        return self._inv_scale_y(out)

    def _postprocess_output(self, y: pd.DataFrame) -> pd.DataFrame:
        """Postprocess outputs. Default to do nothing.

        This method is a placeholder for child classes that may need
        to do specific postprocesing, for example the HierarchicalProphet

        Parameters
        ----------
        y : pd.DataFrame
            dataframe with output predictions

        Returns
        -------
        pd.DataFrame
            postprocessed dataframe

        """
        return y

    def _get_predictive_samples_dict(
        self, fh: ForecastingHorizon, X: Optional[pd.DataFrame] = None
    ) -> dict[str, jnp.ndarray]:
        """
        Return a dictionary of predictive samples for each time series.

        Parameters
        ----------
        fh : ForecastingHorizon or int
            The forecasting horizon specifying the time points to forecast.
        X : array-like, optional (default=None)
            The input features for the time series forecasting model.

        Returns
        -------
        predictive_samples_dict : dict[str, jnp.ndarray]
            A dictionary containing predictive samples for each time series in the input
            data. The keys are the names of the time series, and the values are NumPy
            arrays representing the predictive samples.
        """
        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        predict_data = self._get_predict_data(X=X, fh=fh)

        predictive_samples_ = self.inference_engine_.predict(**predict_data)

        keys_to_delete = []
        for key in predictive_samples_.keys():
            if key.endswith(":ignore"):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del predictive_samples_[key]
        return predictive_samples_

    def predict_all_sites_samples(self, fh, X=None):
        """
        Predicts samples for all sites.

        Parameters
        ----------
        fh : int or array-like
            The forecast horizon or an array-like object representing the forecast
            horizons.
        X : array-like, optional
            The input features for prediction. Defaults to None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the predicted samples for all sites.
        """
        deprecation_warning(
            "predict_all_sites_samples",
            "0.5.0",
            "Use the `predict_component_samples` method instead.",
        )
        return self.predict_component_samples(fh=fh, X=X)

    def predict_component_samples(self, fh, X=None):
        """
        Predicts samples for all sites.

        Parameters
        ----------
        fh : int or array-like
            The forecast horizon or an array-like object representing the forecast
            horizons.
        X : array-like, optional
            The input features for prediction. Defaults to None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the predicted samples for all sites.
        """
        if self._is_vectorized:
            return self._vectorize_predict_method(
                "predict_all_sites_samples", X=X, fh=fh
            )

        predictive_samples_ = self._get_predictive_samples_dict(fh=fh, X=X)

        fh_as_index = self.fh_to_index(fh)
        dfs = []
        for site, data in predictive_samples_.items():

            idxs = self.periodindex_to_multiindex(fh_as_index)

            samples_idx = np.arange(data.shape[0])

            def _coerce_to_tuple(x):
                if isinstance(x, tuple):
                    return x
                return (x,)

            tuples = [
                (sample_i, *_coerce_to_tuple(idx))
                for sample_i, idx in itertools.product(samples_idx, idxs)
            ]
            # Set samples_idx as level 0 of idx
            idx = pd.MultiIndex.from_tuples(tuples, names=["sample", *idxs.names])

            dfs.append(
                pd.DataFrame(
                    data={site: data.flatten()},
                    index=idx,
                )
            )

        df = pd.concat(dfs, axis=1)

        return df

    def predict_samples(self, fh: ForecastingHorizon, X: Optional[pd.DataFrame] = None):
        """
        Generate samples from the posterior predictive distribution.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Exogenous variables.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        pd.DataFrame
            Samples from the posterior predictive distribution.
        """
        if self._is_vectorized:
            return self._vectorize_predict_method("predict_samples", X=X, fh=fh)

        fh_as_index = self.fh_to_index(fh)

        predictive_samples_ = self._get_predictive_samples_dict(fh=fh, X=X)

        observation_site = predictive_samples_["obs"]
        n_samples = predictive_samples_["obs"].shape[0]
        preds = pd.DataFrame(
            data=observation_site.T.reshape((-1, n_samples)),
            columns=list(range(n_samples)),
            index=self.periodindex_to_multiindex(fh_as_index),
        ).sort_index()

        return self._inv_scale_y(preds)

    def _set_y_scales(self, y: pd.DataFrame):
        """
        Set the scaling factor for the target variable.

        * If `scale` attribute is set, it will be used as the scaling factor.
        * If the target variable has only one level, the scaling factor will be set
            as the maximum absolute value of the target variable.
        * If the target variable has multiple levels, the scaling factor will be
            set as the maximum absolute value of each level, grouped by the remaining
            levels.


        Parameters
        ----------
            y (pd.Series or pd.DataFrame): The target variable.

        Returns
        -------
            None
        """
        if self.scale is not None:
            self._scale = self.scale
        elif y.index.nlevels == 1:
            self._scale = float(y.abs().max().values[0])
        else:
            self._scale = y.groupby(level=list(range(y.index.nlevels - 1))).agg(
                lambda x: np.abs(x).max()
            )

        if isinstance(self._scale, (float, int)):
            if self._scale == 0:
                self._scale = 1
        elif isinstance(self._scale, (pd.Series, pd.DataFrame)):
            # Map any values that are 0 to 1
            self._scale = self._scale.replace(0, 1)

    def _scale_y(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the input DataFrame y (divide it by the scaling factor).

        Parameters
        ----------
        y : pd.DataFrame
            The input DataFrame to be inverse scaled.

        Returns
        -------
        pd.DataFrame
            The inverse scaled DataFrame.

        Notes
        -----
        This method takes a DataFrame `y` as input and performs scaling on it.
        If scaling is skipped, the original DataFrame is returned.
        If the scaling factor is a float, each value in the DataFrame is divided by the
        scaling factor.
        If the scaling factor is a DataFrame, the scaling factor for each observation is
        determined based on the index of the input DataFrame `y`.
        The input DataFrame `y` is then divided by the corresponding scaling factor for
        each observation.
        This method assumes that the scaling factor has already been computed and stored
        in the `_scale` attribute of the class.
        """
        if self._likelihood_is_discrete:
            return y

        if isinstance(self._scale, (int, float)):
            return y / self._scale

        scale_for_each_obs = self._scale.loc[y.index.droplevel(-1)].values
        return y / scale_for_each_obs

    def _inv_scale_y(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse scales the input DataFrame y.

        Parameters
        ----------
        y : pd.DataFrame
            The input DataFrame to be inverse scaled.

        Returns
        -------
        pd.DataFrame
            The inverse scaled DataFrame.

        Notes
        -----
        This method takes a DataFrame `y` as input and performs inverse scaling on it.
        If scaling is skipped, the original DataFrame is returned.
        If the scaling factor is a float, each value in the DataFrame is multiplied by
        the scaling factor.
        If the scaling factor is a DataFrame, the scaling factor for each observation
        is determined based on the index of the input DataFrame `y`.
        The input DataFrame `y` is then multiplied by the corresponding scaling factor
        for each observation.
        This method assumes that the scaling factor has already been computed and stored
        in the `_scale` attribute of the class.
        """
        if self._likelihood_is_discrete:
            return y

        if isinstance(self._scale, float):
            return y * self._scale
        scale_for_each_obs = self._scale.loc[y.index.droplevel(-1)].values
        return y * scale_for_each_obs

    def _predict_quantiles(
        self,
        fh: ForecastingHorizon,
        X: Optional[pd.DataFrame],
        alpha: Union[float, List[float]],
    ):
        """
        Generate quantile forecasts for the given forecasting horizon.

        Parameters
        ----------
        fh: ForecastingHorizon
            The Forecasting horizon.
        X: pd.DataFrame
            Exogenous variables dataframe
        alpha: float or List[float])
            Quantile(s) to compute.

        Returns
        -------
        pd.DataFrame
            Quantile forecasts for the forecasting horizon.
        """
        if isinstance(alpha, float):
            alpha = [alpha]

        forecast_samples_ = self.predict_samples(X=X, fh=fh)

        var_names = self._get_varnames()

        int_idx = pd.MultiIndex.from_product([var_names, alpha])

        quantiles = forecast_samples_.quantile(alpha, axis=1).T
        quantiles.columns = int_idx

        return quantiles

    def periodindex_to_multiindex(self, periodindex: pd.PeriodIndex) -> pd.MultiIndex:
        """
        Convert a PeriodIndex to a MultiIndex.

        Parameters
        ----------
        periodindex: pd.PeriodIndex
            PeriodIndex to convert.

        Returns
        -------
        pd.MultiIndex
            Converted MultiIndex.
        """
        if self._y.index.nlevels == 1:
            return periodindex

        series_id_tuples = self.internal_y_indexes_.droplevel(-1).unique().tolist()

        # Check if base_levels 0 is a iterable:
        if not isinstance(series_id_tuples[0], tuple):
            series_id_tuples = [(x,) for x in series_id_tuples]

        series_id_tuples = self._filter_series_tuples(series_id_tuples)

        return pd.Index(
            map(
                lambda x: (*x[0], x[1]),
                itertools.product(series_id_tuples, periodindex),
            ),
            name=self.internal_y_indexes_.names,
        )

    def _filter_series_tuples(self, levels: List[Tuple]) -> List[Tuple]:
        """Filter series levels.

        This method can be overriden by subclasses to filter the series levels.
        See the multivariate model for an example.

        Parameters
        ----------
        levels : List[Tuple]
            The multiindex levels of the dataframe (excluding the last level, which is
            the time index).

        Returns
        -------
        List[Tuple]
            The multiindex levels of the dataframe after filtering.

        """
        return levels

    def fh_to_index(self, fh: ForecastingHorizon, y: pd.DataFrame = None):
        """
        Convert a ForecastingHorizon to an index.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        y : pd.DataFrame, optional
            Target variable. Defaults to None.

        Returns
        -------
        pd.Index
            Index corresponding to the forecasting horizon.
        """
        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        if y is None:
            y = self._y
        fh_dates = fh.to_absolute_index(cutoff=y.index.get_level_values(-1).max())
        return fh_dates

    @property
    def var_names(self):
        """
        Return variable names.

        Returns
        -------
        List[str]
            List of variable names.
        """
        return list(self.distributions_.keys())

    def plot_trace(self, *, figsize=(15, 25), compact=True, var_names=None, **kwargs):
        """
        Plot trace of the MCMC samples using Arviz.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 25).
        compact : bool, optional
            Whether to use compact display. Default is True.
        var_names : list, optional
            List of variable names to plot. Default is None.

        Returns
        -------
        Figure
            Arviz plot trace figure.
        """
        import arviz as az

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
        """
        Return site names.

        Returns
        -------
        List[Any]
            List of site names.
        """
        return list(self.posterior_samples_.keys())

    def _vectorize_predict_method(
        self, methodname: str, X: pd.DataFrame, fh: ForecastingHorizon
    ):
        """
        Handle sktime's "vectorization" of timeseries.

        When a multiindex is passed as input and the forecaster does not accept natively
        hierarchical data, it will create a dataframe of forecasters, and they should be
        called with the corresponding index.

        Parameters
        ----------
        methodname : str
            The method name to call.
        X : pd.DataFrame
            The input data.
        fh : ForecastingHorizon
            The forecasting horizon.

        Returns
        -------
        pd.DataFrame
            The output of the method.
        """
        if not self._is_vectorized:
            return getattr(self, methodname)(X=X, fh=fh)

        def _coerce_to_tuple(x):
            if isinstance(x, (tuple, list)):
                return x
            return (x,)

        outs = []
        for idx, data in self.forecasters_.iterrows():
            forecaster = data[0]

            if X is None:
                _X = None
            else:
                _X = get_multiindex_loc(X, [idx])
                # Keep only index level -1
                for _ in range(_X.index.nlevels - 1):
                    _X = _X.droplevel(0)
            out = getattr(forecaster, methodname)(X=_X, fh=fh)

            if not isinstance(idx, (tuple, list)):
                idx = [idx]
            new_index = pd.MultiIndex.from_tuples(
                [[*idx, *_coerce_to_tuple(dateidx)] for dateidx in out.index]
            )
            out.set_index(new_index, inplace=True)
            outs.append(out)
        return pd.concat(outs, axis=0)


class BaseProphetForecaster(_HeterogenousMetaEstimator, BaseBayesianForecaster):
    """Base class for Bayesian estimators with Effects objects.

    Parameters
    ----------
    trend : Union[str, BaseEffect], optional
        One of "linear" (default), "linear1" or "logistic". Type of trend to use.
        Can also be a custom effect object.

    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, can be positive or negative.
          Absolute value must be less than number of training points.
          The range will be that number of points.
          A negative int indicates number of points
          counting from the end of the history, a positive int from the beginning.

    changepoint_prior_scale : float, optional, default=0.001
        Regularization parameter controlling the flexibility
        of the automatic changepoint selection.

    offset_prior_scale : float, optional, default=0.1
        Scale parameter for the prior distribution of the offset.
        The offset is the constant term in the piecewise trend equation.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    exogenous_effects : List[Tuple[str, BaseEffect, str]]
        List of exogenous effects to apply to the data. Each item of the list
        is a tuple with the name of the effect, the effect object, and the regex
        pattern to match the columns of the dataframe.

    default_effect : Optional[BaseEffect]
        Default effect to apply to the columns that do not match any regex pattern.
        If None, a LinearEffect is used.
    """

    _steps_attr = "_exogenous_effects"
    _steps_fitted_attr = "exogenous_effects_"

    def __init__(
        self,
        trend: Union[BaseEffect, str] = "linear",
        changepoint_interval: int = 25,
        changepoint_range: Union[float, int] = 0.8,
        changepoint_prior_scale: float = 0.001,
        offset_prior_scale: float = 0.1,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        exogenous_effects: Optional[List[BaseEffect]] = None,
        default_effect: Optional[BaseEffect] = None,
        rng_key: jax.typing.ArrayLike = None,
        inference_method: str = "map",
        mcmc_samples: int = 2000,
        mcmc_warmup: int = 200,
        mcmc_chains: int = 4,
        optimizer_steps: int = 100_000,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        inference_engine: Optional[BaseInferenceEngine] = None,
        scale=None,
    ):

        # Trend related hyperparams
        self.trend = trend
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc

        # Exogenous variables related hyperparams
        self.exogenous_effects = exogenous_effects
        self.default_effect = default_effect
        super().__init__(
            rng_key=rng_key,
            inference_method=inference_method,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            optimizer_steps=optimizer_steps,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            scale=scale,
            inference_engine=inference_engine,
        )

        self._trend = self._get_trend_model()

    @property
    def _exogenous_effects(self):
        """Return exogenous effects.

        This property is for compatibility with _HeterogenousMetaEstimator.
        """
        if self.exogenous_effects is None:
            return []
        return [(name, effect) for name, effect, _ in self.exogenous_effects]

    @_exogenous_effects.setter
    def _exogenous_effects(self, value):
        """Set exogenous effects.

        This property is for compatibility with _HeterogenousMetaEstimator.
        """
        # Ensure that user is passing list of (name, effect) or
        # (name, effect, regex) tuples
        assert len(value[0]) in [2, 3], "Invalid value for exogenous_effects"
        len_values = np.all([len(x) == len(value[0]) for x in value])
        assert len_values, "All tuples must have the same length"

        self.exogenous_effects = [
            (name, effect, regex)
            for ((name, effect), (_, _, regex)) in zip(value, self.exogenous_effects)
        ]

    def _fit_effects(
        self, X: Union[None, pd.DataFrame], y: Optional[pd.DataFrame] = None
    ):
        """
        Set custom effects for the features.

        Parameters
        ----------
        feature_names : pd.Index
            List of feature names (obtained with X.columns)
        """
        fitted_effects_list_: List[Tuple[str, BaseEffect, List[str]]] = []
        columns_with_effects: set[str] = set()
        exogenous_effects: Union[List[Tuple[str, BaseEffect, str]], List] = (
            self.exogenous_effects or []
        )

        for effect_name, effect, regex in exogenous_effects:

            if X is not None:
                columns = self._match_columns(X.columns, regex)
                X_columns = X[columns]
            else:
                X_columns = None

            effect = effect.clone()

            effect.fit(  # type: ignore[attr-defined]
                X=X_columns, y=y, scale=self._scale
            )

            if columns_with_effects.intersection(columns):
                msg = "Columns {} are already set".format(
                    columns_with_effects.intersection(columns)
                )

                warnings.warn(msg, UserWarning, stacklevel=2)

            if not len(columns):
                msg = f"No columns match the regex {regex}"
                warnings.warn(msg, UserWarning, stacklevel=2)

            columns_with_effects = columns_with_effects.union(columns)
            fitted_effects_list_.append((effect_name, effect, columns))

        if X is not None:

            features_without_effects: List[str] = X.columns.difference(
                columns_with_effects
            ).tolist()

            if len(features_without_effects) > 0:
                X_columns = X[features_without_effects]

                if self.default_effect is None:
                    default_effect = LinearEffect(
                        prior=dist.Normal(0, 1),
                        effect_mode="additive",
                    )
                else:
                    default_effect = self.default_effect

                default_effect = default_effect.clone()
                default_effect.fit(
                    X=X[features_without_effects],
                    y=y,
                    scale=self._scale,  # type: ignore[attr-defined]
                )
                fitted_effects_list_.append(
                    (
                        "exogenous_variables_effect",
                        default_effect,
                        features_without_effects,
                    )
                )

        self.exogenous_effects_ = fitted_effects_list_

    def _transform_effects(self, X: pd.DataFrame, fh: pd.Index) -> OrderedDict:
        """
        Get exogenous data array.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        fh : pd.Index
            Forecasting horizon as an index.

        Returns
        -------
        dict
            Dictionary of exogenous data arrays.
        """
        out = OrderedDict()
        for effect_name, effect, columns in self.exogenous_effects_:
            # If no columns are found, skip
            if columns is None or len(columns) == 0:
                if effect.get_tag("skip_predict_if_no_match"):
                    continue

            data: Dict[str, jnp.ndarray] = effect.transform(X[columns], fh=fh)
            out[effect_name] = data

        return out

    @property
    def non_skipped_exogenous_effect(self) -> dict[str, BaseEffect]:
        """
        Return exogenous effect dictionary.

        Returns
        -------
        dict
            Dictionary of exogenous effects.
        """
        return {
            effect_name: effect
            for effect_name, effect, columns in self.exogenous_effects_
            if len(columns) > 0 or not effect.get_tag("skip_predict_if_no_match")
        }

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
        if isinstance(self.trend, str):
            deprecation_warning(
                "trend (str)",
                "0.5.0",
                "Pass a BaseEffect instance instead.",
            )
        # Changepoints and trend
        if self.trend == "linear":
            return PiecewiseLinearTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
            )

        elif self.trend == "linear_raw":
            return PiecewiseLinearTrend(
                changepoint_interval=self.changepoint_interval,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                offset_prior_scale=self.offset_prior_scale,
                remove_seasonality_before_suggesting_initial_vals=False,
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

    def _validate_hyperparams(self):
        """Validate the hyperparameters."""
        if self.changepoint_interval <= 0:
            raise ValueError("changepoint_interval must be greater than 0.")
        if self.changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be greater than 0.")
        if self.capacity_prior_scale <= 0:
            raise ValueError("capacity_prior_scale must be greater than 0.")
        if self.capacity_prior_loc <= 0:
            raise ValueError("capacity_prior_loc must be greater than 0.")
        if self.offset_prior_scale <= 0:
            raise ValueError("offset_prior_scale must be greater than 0.")
        if self.trend not in [
            "linear",
            "linear_raw",
            "logistic",
            "flat",
        ] and not isinstance(self.trend, BaseEffect):
            raise ValueError('trend must be either "linear" or "logistic".')

    def _match_columns(
        self, columns: Union[pd.Index, List[str]], regex: Union[str, None]
    ) -> pd.Index:
        """Match the columns of the DataFrame with the regex pattern.

        Parameters
        ----------
        columns : pd.Index
            Columns of the dataframe.

        Returns
        -------
        pd.Index
            The columns that match the regex pattern.

        Raises
        ------
        ValueError
            Indicates the abscence of required regex pattern.
        """
        if isinstance(columns, List):
            columns = pd.Index(columns)

        if regex is None:
            return []

        columns = columns.astype(str)
        return columns[columns.str.match(regex)]

    def __rshift__(self, other):
        """Right shift operator.

        This method allows to chain effects and inference engines using the right shift

        Paremeters
        ----------
        other : BaseEffect, tuple, list or BaseInferenceEngine
            The object to chain with the current object.

        Returns
        -------
        BaseProphetForecaster
            A new instance of the class with the specified effect or inference engine.

        Raises
        ------
        ValueError
            If the type of the object is not valid.
        """
        exogenous_effects = (
            [] if self.exogenous_effects is None else self.exogenous_effects
        )
        if isinstance(other, BaseEffect):
            new_obj = self.clone()
            return new_obj.set_params(trend=other)
        if isinstance(other, tuple):

            assert len(other) == 3
            assert isinstance(other[1], BaseEffect)

            new_obj = self.clone()
            return new_obj.set_params(exogenous_effects=[*exogenous_effects, other])

        if isinstance(other, list):
            new_obj = self.clone()
            return new_obj.set_params(exogenous_effects=[*exogenous_effects, *other])

        if isinstance(other, BaseInferenceEngine):
            new_obj = self.clone()
            return new_obj.set_params(inference_engine=other)

        raise ValueError("Invalid type for right shift operator")

    def _sk_visual_block_(self):
        """
        Implement visual block.

        The try except is used since the methods used inside
        _get_default_visual_block are private in sktime, and not public,
        and may be changed in the future.
        """
        try:
            # Tries to get the default visual block
            return self._get_default_visual_block()

        except Exception as e:
            return super()._sk_visual_block_()

    def _get_default_visual_block(self):
        """Make default visual block."""
        from sktime.utils._estimator_html_repr import _VisualBlock, _get_visual_block

        visual_blocks = []
        visual_block_names = []

        steps = getattr(self, self._steps_attr)
        steps = [("trend", self._trend)] + steps
        names, estimators = zip(*steps)

        name_details = [str(est) for est in estimators]

        visual_blocks.append(
            _VisualBlock(
                "serial",
                estimators,
                names=names,
                name_details=name_details,
                dash_wrapped=False,
            )
        )
        visual_block_names.append("effects")

        inference_engine = self._inference_engine
        visual_blocks.append(_get_visual_block(inference_engine))
        visual_block_names.append("inference_engine")

        return _VisualBlock(
            "parallel",
            visual_blocks,
            names=visual_block_names,
            name_details=[None] * len(visual_blocks),
            dash_wrapped=False,
        )
