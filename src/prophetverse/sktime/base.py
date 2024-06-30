"""Base classes for sktime forecasters in prophetverse."""

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


from prophetverse.effects import AbstractEffect, LinearEffect
from prophetverse.engine import MAPInferenceEngine, MCMCInferenceEngine
from prophetverse.utils import get_multiindex_loc, series_to_tensor


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
        rng_key: jax.typing.ArrayLike,
        inference_method: str,
        mcmc_samples: int,
        mcmc_warmup: int,
        mcmc_chains: int,
        optimizer_steps: int,
        optimizer_name: str,
        optimizer_kwargs: dict,
        scale=None,
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
        super().__init__(*args, **kwargs)

    @property
    def should_skip_scaling(self):
        """Property that indicates whether the forecaster uses a discrete likelihood.

        As a consequence, the target variable must be integer-valued and will not be
        scaled before _get_fit_data is called.

        Returns
        -------
        bool
            True if the forecaster uses a discrete likelihood, False otherwise.
        """
        return False

    def optimizer(self) -> numpyro.optim._NumPyroOptim:
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

            import optax
            from numpyro.optim import optax_to_numpyro

            scheduler = optax.cosine_decay_schedule(**optimizer_kwargs)

            opt = optax_to_numpyro(
                optax.chain(
                    optax.scale_by_adam(),
                    optax.scale_by_schedule(scheduler),
                    optax.scale(-1.0),
                )
            )
            return opt

        return getattr(numpyro.optim, optimizer_name)(**optimizer_kwargs)

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

        if self.inference_method == "mcmc":
            self.inference_engine_ = MCMCInferenceEngine(
                self.model,
                num_samples=self.mcmc_samples,
                num_warmup=self.mcmc_warmup,
                num_chains=self.mcmc_chains,
                rng_key=rng_key,
            )
        elif self.inference_method == "map":
            self.inference_engine_ = MAPInferenceEngine(
                self.model,
                rng_key=rng_key,
                optimizer_factory=self.optimizer,
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
        predictive_samples = self.predict_samples(fh=fh, X=X)
        y_pred = predictive_samples.mean(axis=1).to_frame(self._y.columns[0])

        return y_pred

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
        if self.should_skip_scaling:
            return y

        if isinstance(self._scale, float):
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
        if self.should_skip_scaling:
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
            outs.append(out)
        return pd.concat(outs, axis=0)


class ExogenousEffectMixin:
    """
    Mixin that handles the connection between exogenous effects and input data.

    Parameters
    ----------
    exogenous_effects : List[AbstractEffect]
        List of exogenous effects.
    default_effect : AbstractEffect, optional
        Default effect to apply. Default is None.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self, exogenous_effects: List[AbstractEffect], default_effect=None, **kwargs
    ):

        self.exogenous_effects = exogenous_effects
        self.default_effect = default_effect
        super().__init__(**kwargs)

    def _set_custom_effects(self, feature_names: pd.Index):
        """
        Set custom effects for the features.

        Parameters
        ----------
        feature_names : pd.Index
            List of feature names (obtained with X.columns)
        """
        effects_and_columns: Dict[str, Tuple[set[str], AbstractEffect]] = {}
        columns_with_effects: set[str] = set()
        exogenous_effects: Union[list[AbstractEffect], set[Any]] = (
            self.exogenous_effects or set()
        )

        default_effect = self.default_effect
        if default_effect is None:
            default_effect = LinearEffect(
                id="exogenous_variables_effect",
                prior=dist.Normal(0, 1),
                effect_mode="additive",
            )

        for effect in exogenous_effects:

            columns = effect.match_columns(feature_names)

            if columns_with_effects.intersection(columns):
                raise ValueError(
                    "Columns {} are already set".format(
                        columns_with_effects.intersection(columns)
                    )
                )

            if not len(columns):
                logging.warning(f"No columns match the regex {effect.regex}")

            columns_with_effects = columns_with_effects.union(columns)

            effects_and_columns.update(
                {
                    effect.id: (
                        columns,
                        effect,
                    )
                }
            )

        features_without_effects: set = feature_names.difference(columns_with_effects)

        if len(features_without_effects):

            effects_and_columns.update(
                {
                    "exogenous_variables_effect": (
                        features_without_effects,
                        default_effect,
                    )
                }
            )

        self._exogenous_effects_and_columns = effects_and_columns

    def _get_exogenous_data_array(self, X: pd.DataFrame):
        """
        Get exogenous data array.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        dict
            Dictionary of exogenous data arrays.
        """
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
        """
        Return exogenous effect dictionary.

        Returns
        -------
        dict
            Dictionary of exogenous effects.
        """
        return {
            k: v[1] for k, v in self._exogenous_effects_and_columns.items() if len(v[0])
        }
