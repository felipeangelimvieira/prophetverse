"""Univariate Prophet model.

This module implements the Univariate Prophet model, similar to the one implemented in
the `prophet` library.

"""

import jax.numpy as jnp
import pandas as pd
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon

from prophetverse.models import (
    univariate_gamma_model,
    univariate_model,
    univariate_negbinomial_model,
)
from prophetverse.sktime.base import BaseEffectsBayesianForecaster, Stage
from prophetverse.trend.flat import FlatTrend
from prophetverse.trend.piecewise import (
    PiecewiseLinearTrend,
    PiecewiseLogisticTrend,
    TrendModel,
)

__all__ = ["Prophetverse", "Prophet", "ProphetGamma", "ProphetNegBinomial"]


_LIKELIHOOD_MODEL_MAP = {
    "normal": univariate_model,
    "gamma": univariate_gamma_model,
    "negbinomial": univariate_negbinomial_model,
}

_DISCRETE_LIKELIHOODS = ["negbinomial"]


class Prophetverse(BaseEffectsBayesianForecaster):
    """Univariate ``Prophetverse`` forecaster, with support for multiple likelihoods.

    Differences to facebook's prophet:

    * logistic trend. Here, another parametrization is considered,
      and the capacity is not passed as input, but inferred from the data.

    * the users can pass arbitrary ``sktime`` transformers as ``feature_transformer``,
      for instance ``FourierFeatures`` or ``HolidayFeatures``.

    * no default weekly_seasonality/yearly_seasonality, this is left to the user
      via the ``feature_transformer`` parameter

    * Uses ``changepoint_interval`` instead of ``n_changepoints`` to set changepoints.

    * accepts configurations where each exogenous variable has a different function
      relating it to its additive effect on the time series.
      One can, for example, set different priors for a group of feature,
      or use a Hill function to model the effect of a feature.

    Parameters
    ----------
    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, ca nbe positive or negative.
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

    feature_transformer : sktime transformer, BaseTransformer, optional, default=None
        Transformer object to generate Fourier terms, holiday or other features.
        If None, no additional features are used.
        For multiple features, pass a ``FeatureUnion`` object with the transformers.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    noise_scale : float, optional, default=0.05
        Scale parameter for the observation noise.
    trend : str, optional, one of "linear" (default) or "logistic"
        Type of trend to use. Can be "linear" or "logistic".

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of MCMC warmup steps. Also known as burn-in.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains to run in parallel.

    inference_method : str, optional, one of "mcmc" or "map", default="map"
        Inference method to use. Can be "mcmc" or "map".

    optimizer_name : str, optional, default="Adam"
        Name of the numpyro optimizer to use for variational inference.

    optimizer_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the numpyro optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps to perform for variational inference.

    exogenous_effects : List[AbstractEffect], optional, default=None
        A list of ``prophetverse`` ``AbstractEffect`` objects
        defining the exogenous effects to be used in the model.

    likelihood : str, optional, default="normal"
        Likelihood to use for the model. Can be "normal", "gamma" or "negbinomial".

    default_effect : AbstractEffectm optional, defalut=None
        The default effect to be used when no effect is specified for a variable.

    default_exogenous_prior : tuple, default=None
        Default prior distribution for exogenous effects.

    rng_key : jax.random.PRNGKey or None (default
        Random number generator key.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        "python_dependencies": "prophetverse",
        # estimator type
        # --------------
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "enforce_index_type": [pd.Period, pd.DatetimeIndex],
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="linear",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        likelihood="normal",
        default_effect=None,
        scale=None,
        rng_key=None,
    ):
        """Initialize the Prophet model."""
        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.noise_scale = noise_scale
        self.feature_transformer = feature_transformer
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.trend = trend
        self.likelihood = likelihood

        super().__init__(
            rng_key=rng_key,
            # ExogenousEffectMixin
            default_effect=default_effect,
            exogenous_effects=exogenous_effects,
            # BaseBayesianForecaster
            inference_method=inference_method,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            scale=scale,
        )

        self._validate_hyperparams()

    @property
    def model(self):
        """Return the model function.

        Returns
        -------
        Callable
            A function that can be passed to Numpyro samplers.
        """
        return _LIKELIHOOD_MODEL_MAP[self.likelihood]

    @property
    def should_skip_scaling(self) -> bool:
        """Skip scaling if the likelihood is discrete.

        In the case of discrete likelihoods, the data is not scaled since this can
        create non-integer values.
        """
        return self.likelihood in _DISCRETE_LIKELIHOODS

    def _validate_hyperparams(self):
        """Validate the hyperparameters."""
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
        if self.trend not in ["linear", "logistic", "flat"] and not isinstance(
            self.trend, TrendModel
        ):
            raise ValueError('trend must be either "linear" or "logistic".')

        if self.likelihood not in _LIKELIHOOD_MODEL_MAP:
            raise ValueError(
                f"likelihood must be one of {list(_LIKELIHOOD_MODEL_MAP.keys())}"
                + "Got {self.likelihood}."
            )

    def _get_fit_data(self, y, X, fh):
        """
        Prepare the data for the Numpyro model.

        Parameters
        ----------
        y: pd.DataFrame
            Time series data.
        X: pd.DataFrame
            Exogenous variables.
        fh: ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        dict
            Dictionary of data for the Numpyro model.
        """
        fh = y.index.get_level_values(-1).unique()

        self.trend_model_ = self._get_trend_model()

        if self.should_skip_scaling:
            self.trend_model_.initialize(y / self._scale)
        else:
            self.trend_model_.initialize(y)

        trend_data = self.trend_model_.fit(fh)

        # Exogenous features

        if X is None:
            X = pd.DataFrame(index=y.index)

        if self.feature_transformer is not None:

            X = self.feature_transformer.fit_transform(X)

        self._has_exogenous = ~X.columns.empty
        X = X.loc[y.index]

        self._fit_effects(X)
        exogenous_data = self._transform_effects(X, stage=Stage.TRAIN)

        y_array = jnp.array(y.values.flatten()).reshape((-1, 1))

        # Inputs that also are used in predict
        self.fit_and_predict_data_ = {
            "trend_model": self.trend_model_,
            "noise_scale": self.noise_scale,
            "scale": self._scale,
            "exogenous_effects": (
                self.non_skipped_exogenous_effect if self._has_exogenous else None
            ),
        }

        inputs = {
            "y": y_array,
            "data": exogenous_data,
            "trend_data": trend_data,
            **self.fit_and_predict_data_,
        }

        return inputs

    def _get_predict_data(self, X: pd.DataFrame, fh: ForecastingHorizon) -> dict:
        """
        Prepare the data for making predictions.

        Parameters
        ----------
        X: pd.DataFrame
            Exogenous variables.
        fh: ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        dict
            Dictionary of data for the Numpyro model.
        """
        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        trend_data = self.trend_model_.fit(fh_as_index)

        if X is None:
            X = pd.DataFrame(index=fh_as_index)

        if self.feature_transformer is not None:
            X = self.feature_transformer.transform(X)

        exogenous_data = (
            self._transform_effects(X.loc[fh_as_index], stage=Stage.PREDICT)
            if self._has_exogenous
            else None
        )

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
        TrendModel
            The trend model based on the specified trend parameter.

        Raises
        ------
        ValueError
            If the trend parameter is not one of 'linear', 'logistic', 'flat'
            or a TrendModel instance.
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

        elif isinstance(self.trend, TrendModel):
            return self.trend

        raise ValueError(
            "trend must be either 'linear', 'logistic' or a TrendModel instance."
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # pragma: no cover
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
        params = []
        for likelihood in _LIKELIHOOD_MODEL_MAP.keys():
            params.append(
                {
                    "likelihood": likelihood,
                    "optimizer_steps": 10,
                }
            )
        return params


class Prophet(Prophetverse):
    """Univariate ``Prophetverse`` forecaster - prophet model implemented in numpyro.

    Differences to facebook's prophet:

    * logistic trend. Here, another parametrization is considered,
      and the capacity is not passed as input, but inferred from the data.

    * the users can pass arbitrary ``sktime`` transformers as ``feature_transformer``,
      for instance ``FourierFeatures`` or ``HolidayFeatures``.

    * no default weekly_seasonality/yearly_seasonality, this is left to the user
      via the ``feature_transformer`` parameter

    * Uses ``changepoint_interval`` instead of ``n_changepoints`` to set changepoints.

    * accepts configurations where each exogenous variable has a different function
      relating it to its additive effect on the time series.
      One can, for example, set different priors for a group of feature,
      or use a Hill function to model the effect of a feature.

    Parameters
    ----------
    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, ca nbe positive or negative.
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

    feature_transformer : sktime transformer, BaseTransformer, optional, default=None
        Transformer object to generate Fourier terms, holiday or other features.
        If None, no additional features are used.
        For multiple features, pass a ``FeatureUnion`` object with the transformers.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    noise_scale : float, optional, default=0.05
        Scale parameter for the observation noise.
    trend : str, optional, one of "linear" (default) or "logistic"
        Type of trend to use. Can be "linear" or "logistic".

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of MCMC warmup steps. Also known as burn-in.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains to run in parallel.

    inference_method : str, optional, one of "mcmc" or "map", default="map"
        Inference method to use. Can be "mcmc" or "map".

    optimizer_name : str, optional, default="Adam"
        Name of the numpyro optimizer to use for variational inference.

    optimizer_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the numpyro optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps to perform for variational inference.

    exogenous_effects : List[AbstractEffect], optional, default=None
        A list of ``prophetverse`` ``AbstractEffect`` objects
        defining the exogenous effects to be used in the model.

    default_effect : AbstractEffectm optional, defalut=None
        The default effect to be used when no effect is specified for a variable.

    default_exogenous_prior : tuple, default=None
        Default prior distribution for exogenous effects.

    rng_key : jax.random.PRNGKey or None (default
        Random number generator key.
    """

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="logistic",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            feature_transformer=feature_transformer,
            capacity_prior_scale=capacity_prior_scale,
            capacity_prior_loc=capacity_prior_loc,
            noise_scale=noise_scale,
            trend=trend,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            exogenous_effects=exogenous_effects,
            likelihood="normal",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
        )


class ProphetGamma(Prophetverse):
    """Gamma-likelihood prophet.

    Parameters
    ----------
    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, ca nbe positive or negative.
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

    feature_transformer : sktime transformer, BaseTransformer, optional, default=None
        Transformer object to generate Fourier terms, holiday or other features.
        If None, no additional features are used.
        For multiple features, pass a ``FeatureUnion`` object with the transformers.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    noise_scale : float, optional, default=0.05
        Scale parameter for the observation noise.
    trend : str, optional, one of "linear" (default) or "logistic"
        Type of trend to use. Can be "linear" or "logistic".

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of MCMC warmup steps. Also known as burn-in.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains to run in parallel.

    inference_method : str, optional, one of "mcmc" or "map", default="map"
        Inference method to use. Can be "mcmc" or "map".

    optimizer_name : str, optional, default="Adam"
        Name of the numpyro optimizer to use for variational inference.

    optimizer_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the numpyro optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps to perform for variational inference.

    exogenous_effects : List[AbstractEffect], optional, default=None
        A list of ``prophetverse`` ``AbstractEffect`` objects
        defining the exogenous effects to be used in the model.

    default_effect : AbstractEffectm optional, defalut=None
        The default effect to be used when no effect is specified for a variable.

    default_exogenous_prior : tuple, default=None
        Default prior distribution for exogenous effects.

    rng_key : jax.random.PRNGKey or None (default
        Random number generator key.
    """

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="logistic",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            feature_transformer=feature_transformer,
            capacity_prior_scale=capacity_prior_scale,
            capacity_prior_loc=capacity_prior_loc,
            noise_scale=noise_scale,
            trend=trend,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            exogenous_effects=exogenous_effects,
            likelihood="gamma",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
        )


class ProphetNegBinomial(Prophetverse):
    """Univariate ``Prophetverse`` forecaster with negative binomial likelihood.

    Parameters
    ----------
    changepoint_interval : int, optional, default=25
        Number of potential changepoints to sample in the history.

    changepoint_range : float or int, optional, default=0.8
        Proportion of the history in which trend changepoints will be estimated.

        * if float, must be between 0 and 1.
          The range will be that proportion of the training history.

        * if int, ca nbe positive or negative.
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

    feature_transformer : sktime transformer, BaseTransformer, optional, default=None
        Transformer object to generate Fourier terms, holiday or other features.
        If None, no additional features are used.
        For multiple features, pass a ``FeatureUnion`` object with the transformers.

    capacity_prior_scale : float, optional, default=0.2
        Scale parameter for the prior distribution of the capacity.

    capacity_prior_loc : float, optional, default=1.1
        Location parameter for the prior distribution of the capacity.

    noise_scale : float, optional, default=0.05
        Scale parameter for the observation noise.
    trend : str, optional, one of "linear" (default) or "logistic"
        Type of trend to use. Can be "linear" or "logistic".

    mcmc_samples : int, optional, default=2000
        Number of MCMC samples to draw.

    mcmc_warmup : int, optional, default=200
        Number of MCMC warmup steps. Also known as burn-in.

    mcmc_chains : int, optional, default=4
        Number of MCMC chains to run in parallel.

    inference_method : str, optional, one of "mcmc" or "map", default="map"
        Inference method to use. Can be "mcmc" or "map".

    optimizer_name : str, optional, default="Adam"
        Name of the numpyro optimizer to use for variational inference.

    optimizer_kwargs : dict, optional, default={}
        Additional keyword arguments to pass to the numpyro optimizer.

    optimizer_steps : int, optional, default=100_000
        Number of optimization steps to perform for variational inference.

    exogenous_effects : List[AbstractEffect], optional, default=None
        A list of ``prophetverse`` ``AbstractEffect`` objects
        defining the exogenous effects to be used in the model.

    default_effect : AbstractEffect optional, defalut=None
        The default effect to be used when no effect is specified for a variable.

    default_exogenous_prior : tuple, default=None
        Default prior distribution for exogenous effects.

    rng_key : jax.random.PRNGKey or None (default
        Random number generator key.
    """

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        offset_prior_scale=0.1,
        feature_transformer=None,
        capacity_prior_scale=0.2,
        capacity_prior_loc=1.1,
        noise_scale=0.05,
        trend="logistic",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        inference_method="map",
        optimizer_name="Adam",
        optimizer_kwargs=None,
        optimizer_steps=100_000,
        exogenous_effects=None,
        default_effect=None,
        scale=None,
        rng_key=None,
    ):

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            feature_transformer=feature_transformer,
            capacity_prior_scale=capacity_prior_scale,
            capacity_prior_loc=capacity_prior_loc,
            noise_scale=noise_scale,
            trend=trend,
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
            inference_method=inference_method,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_steps=optimizer_steps,
            exogenous_effects=exogenous_effects,
            likelihood="negbinomial",
            default_effect=default_effect,
            scale=scale,
            rng_key=rng_key,
        )
