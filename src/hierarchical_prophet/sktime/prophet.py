import logging
import re
from functools import partial
from collections import OrderedDict
from sktime.transformations.series.detrend import Detrender
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import lax, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.fourier import FourierFeatures
from hierarchical_prophet._utils import convert_index_to_days_since_epoch
from hierarchical_prophet.base import BaseBayesianForecaster, init_params
from hierarchical_prophet.sktime.utils.exogenous_priors import (
    get_exogenous_priors,
    sample_exogenous_coefficients,
)
from hierarchical_prophet.jax_functions import (
    additive_mean_model,
    multiplicative_mean_model,
)

from hierarchical_prophet.logistic import suggest_logistic_rate_and_offset

from hierarchical_prophet.univariate.model import model
from hierarchical_prophet.univariate.trend_utils import (
    get_changepoint_matrix,
    get_changepoint_timeindexes,
)
import functools

logger = logging.getLogger("hierarchical-prophet")

NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000


class Prophet(BaseBayesianForecaster):
    """
    Prophet is a Bayesian time series forecasting model based on the hierarchical-prophet library.

    Args:
        n_changepoints (int): Number of changepoints to be considered in the model.
        changepoint_range (float): Proportion of the data range in which changepoints will be considered.
        changepoint_prior_scale (float): Scale parameter for the Laplace prior distribution of the changepoints.
        growth_offset_prior_scale (float): Scale parameter for the prior distribution of the growth offset.
        capacity_prior_scale (float): Scale parameter for the prior distribution of the capacity.
        capacity_prior_loc (float): Location parameter for the prior distribution of the capacity.
        noise_scale (float): Scale parameter for the observation noise.
        trend (str): Type of trend to be considered in the model. Options are "linear" and "logistic".
        seasonality_mode (str): Mode of seasonality to be considered in the model. Options are "additive" and "multiplicative".
        mcmc_samples (int): Number of MCMC samples to be drawn.
        mcmc_warmup (int): Number of MCMC warmup steps.
        mcmc_chains (int): Number of MCMC chains.
        exogenous_priors (dict): Dictionary specifying the prior distributions for the exogenous variables.
        default_exogenous_prior (tuple): Default prior distribution for the exogenous variables.
        rng_key (jax.random.PRNGKey): Random number generator key.

    Attributes:
        n_changepoints (int): Number of changepoints to be considered in the model.
        changepoint_range (float): Proportion of the data range in which changepoints will be considered.
        changepoint_prior_scale (float): Scale parameter for the Laplace prior distribution of the changepoints.
        noise_scale (float): Scale parameter for the observation noise.
        growth_offset_prior_scale (float): Scale parameter for the prior distribution of the growth offset.
        capacity_prior_scale (float): Scale parameter for the prior distribution of the capacity.
        capacity_prior_loc (float): Location parameter for the prior distribution of the capacity.
        seasonality_mode (str): Mode of seasonality to be considered in the model. Options are "additive" and "multiplicative".
        trend (str): Type of trend to be considered in the model. Options are "linear" and "logistic".
        exogenous_priors (dict): Dictionary specifying the prior distributions for the exogenous variables.
        default_exogenous_prior (tuple): Default prior distribution for the exogenous variables.
        rng_key (jax.random.PRNGKey): Random number generator key.
        _ref_date (None): Reference date for the time series.
        _linear_global_rate (float): Global rate of the linear trend.
        _linear_offset (float): Offset of the linear trend.
        _changepoint_t (jax.numpy.ndarray): Array of changepoint times.
        _changepoint_dists (list): List of prior distributions for the changepoints.
        _exogenous_dists (list): List of prior distributions for the exogenous variables.
        _exogenous_permutation_matrix (jax.numpy.ndarray): Permutation matrix for the exogenous variables.
        _exogenous_coefficients (jax.numpy.ndarray): Coefficients for the exogenous variables.
        _changepoint_coefficients (jax.numpy.ndarray): Coefficients for the changepoints.
        _linear_offset_coef (float): Coefficient for the linear offset.
        _capacity (float): Capacity parameter for the logistic trend.
        _samples_predictive (dict): Dictionary of predictive samples.

    """

    _tags = {
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        changepoint_interval=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.001,
        yearly_seasonality=False,
        weekly_seasonality=False,
        growth_offset_prior_scale=1,
        capacity_prior_scale=None,
        capacity_prior_loc=1,
        noise_scale=0.05,
        trend="linear",
        seasonality_mode="multiplicative",
        mcmc_samples=2000,
        mcmc_warmup=200,
        mcmc_chains=4,
        exogenous_priors=None,
        default_exogenous_prior=(dist.Normal, 0, 1),
        rng_key=random.PRNGKey(24),
    ):
        """
        Initializes a Prophet object.

        Args:
            n_changepoints (int): Number of changepoints to be considered in the model.
            changepoint_range (float): Proportion of the data range in which changepoints will be considered.
            changepoint_prior_scale (float): Scale parameter for the Laplace prior distribution of the changepoints.
            growth_offset_prior_scale (float): Scale parameter for the prior distribution of the growth offset.
            capacity_prior_scale (float): Scale parameter for the prior distribution of the capacity.
            capacity_prior_loc (float): Location parameter for the prior distribution of the capacity.
            noise_scale (float): Scale parameter for the observation noise.
            trend (str): Type of trend to be considered in the model. Options are "linear" and "logistic".
            seasonality_mode (str): Mode of seasonality to be considered in the model. Options are "additive" and "multiplicative".
            mcmc_samples (int): Number of MCMC samples to be drawn.
            mcmc_warmup (int): Number of MCMC warmup steps.
            mcmc_chains (int): Number of MCMC chains.
            exogenous_priors (dict): Dictionary specifying the prior distributions for the exogenous variables.
            default_exogenous_prior (tuple): Default prior distribution for the exogenous variables.
            rng_key (jax.random.PRNGKey): Random number generator key.
        """

        self.changepoint_interval = changepoint_interval
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.noise_scale = noise_scale
        self.growth_offset_prior_scale = growth_offset_prior_scale
        self.capacity_prior_scale = capacity_prior_scale
        self.capacity_prior_loc = capacity_prior_loc
        self.seasonality_mode = seasonality_mode
        self.trend = trend
        self.exogenous_priors = exogenous_priors
        self.default_exogenous_prior = default_exogenous_prior

        super().__init__(
            rng_key=rng_key,
            method="mcmc",
            mcmc_samples=mcmc_samples,
            mcmc_warmup=mcmc_warmup,
            mcmc_chains=mcmc_chains,
        )

        # Define all attributes that are created outside init

        self.t_start = None
        self.t_scale = None
        self.y_scale = None
        self._samples_predictive = None
        self.fourier_feature_transformer_ = None
        self.extra_inputs_ = None
        self.exogenous_columns_ = None
        self.model = model

    def _get_numpyro_model_data(self, y, X, fh):
        """
        Prepares the data for the Numpyro model.

        Args:
            y (pd.DataFrame): Time series data.
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of data for the Numpyro model.
        """
        if X is None or X.columns.empty:
            self._has_exogenous = False
        else:
            self._has_exogenous = True

        self._set_time_and_y_scales(y)
        self._replace_hyperparam_nones_with_defaults(y)
        self._set_changepoints_t(y)
        t = self._index_to_scaled_timearray(y.index)
        changepoint_matrix = self._get_changepoint_matrix(t)

        # Must create empty X
        if not self._has_exogenous and self.has_seasonality:
            X = pd.DataFrame(index=y.index)
            self._has_exogenous = True

        if self.has_seasonality:
            self.init_seasonalities(y, X)
            X = self.add_seasonalities(X)

        self.exogenous_columns_ = X.columns if self._has_exogenous else []

        X_array = (
            jnp.array(X.loc[y.index].values)
            if self.has_exogenous_or_seasonality
            else None
        )
        y_array = jnp.array(y.values.flatten())

        self.extra_inputs_ = self._set_prior_distributions(y, X)
        self.extra_inputs_["trend_mode"] = self.trend
        self.extra_inputs_["seasonality_mode"] = self.seasonality_mode
        self.extra_inputs_["y_scale"] = self.y_scale

        inputs = {
            "t": self._index_to_scaled_timearray(y.index),
            "X": X_array,
            "y": y_array,
            "changepoint_matrix": changepoint_matrix,
            **self.extra_inputs_,
        }

        return inputs
    

    def init_seasonalities(self, y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        sp_list = []
        fourier_term_list = []

        index: pd.PeriodIndex = y.index

        if isinstance(self.yearly_seasonality, bool):
            yearly_seasonality_num_terms = 10
        elif isinstance(self.yearly_seasonality, int):
            yearly_seasonality_num_terms = self.yearly_seasonality
        else:
            raise ValueError("yearly_seasonality must be a boolean or an integer")
        if self.yearly_seasonality:
            sp_list.append("Y")
            fourier_term_list.append(yearly_seasonality_num_terms)

        if isinstance(self.weekly_seasonality, bool):
            weekly_seasonality_num_terms = 3
        elif isinstance(self.weekly_seasonality, int):
            weekly_seasonality_num_terms = self.weekly_seasonality
        else:
            raise ValueError("weekly_seasonality must be a boolean or an integer")

        if self.weekly_seasonality:
            sp_list.append("W")
            fourier_term_list.append(weekly_seasonality_num_terms)

        self.fourier_feature_transformer_ = FourierFeatures(
            sp_list=sp_list, fourier_terms_list=fourier_term_list, freq=index.freq
        ).fit(y)

    def add_seasonalities(self, X):
        return self.fourier_feature_transformer_.transform(X)

    @property
    def has_seasonality(self):
        return self.yearly_seasonality or self.weekly_seasonality

    @property
    def has_exogenous_or_seasonality(self):
        return self._has_exogenous or self.has_seasonality
    
    
    def _get_trend_sample_func(self, y : pd.DataFrame, X: pd.DataFrame):
        t_scaled = self._index_to_scaled_timearray(y.index)
        distributions = {}

        changepoints_loc = jnp.zeros(len(self._changepoint_t))
        detrender = Detrender()
        trend = y - detrender.fit_transform(y)

        if self.trend == "linear":

            linear_global_rate = (trend.values[-1, 0] - trend.values[0, 0]) / (
                t_scaled[-1] - t_scaled[0]
            )
            changepoints_loc.at[0].set(linear_global_rate)

            distributions["changepoint_coefficients"] = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t))
                * (self.changepoint_prior_scale * self.y_scale),
            )

            distributions["offset"] = dist.Normal(
                (trend.values[0, 0] - linear_global_rate * t_scaled[0]),
                0.1 * self.y_scale,
            )

        if self.trend == "logistic":

            linear_global_rate, timeoffset = suggest_logistic_rate_and_offset(
                t_scaled,
                trend.values.flatten(),
                capacities=self.capacity_prior_loc * self.y_scale,
            )

            linear_global_rate = linear_global_rate[0]
            timeoffset = timeoffset[0]
            changepoints_loc.at[0].set(linear_global_rate)

            changepoint_coefficients_distribution = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t)) * self.changepoint_prior_scale,
            )

            distributions["changepoint_coefficients"] = (
                changepoint_coefficients_distribution
            )
            distributions["offset"] = dist.Normal(timeoffset, jnp.log(self.y_scale))

            distributions["capacity"] = dist.Normal(
                self.capacity_prior_loc,
                self.capacity_prior_scale,
            )
            
            
        def init_trend_params(distributions) -> dict:
            
            return init_params(distributions)
        
        return functools.partial(init_trend_params, distributions=distributions)

    def _set_prior_distributions(self, y: pd.DataFrame, X: pd.DataFrame):
        """
        Sets the prior distributions for the model parameters.

        Args:
            X (pd.DataFrame): Time series exogenous data.
        """

        # Get prior distributions from self.exogenous_priors and self.default_exogenous_prior
        # The permutation matrix maps the distributions to the columns of X
        t_scaled = self._index_to_scaled_timearray(y.index)
        distributions = {}
        extra_inputs = {}

        changepoints_loc = jnp.zeros(len(self._changepoint_t))

        detrender = Detrender()
        trend = y - detrender.fit_transform(y)

        if self.trend == "linear":

            linear_global_rate = (trend.values[-1, 0] - trend.values[0, 0]) / (
                t_scaled[-1] - t_scaled[0]
            )
            changepoints_loc.at[0].set(linear_global_rate)

            distributions["changepoint_coefficients"] = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t))
                * (self.changepoint_prior_scale * self.y_scale),
            )

            distributions["offset"] = dist.Normal(
                (trend.values[0, 0] - linear_global_rate * t_scaled[0]),
                0.1 * self.y_scale,
            )

        if self.trend == "logistic":

            linear_global_rate, timeoffset = suggest_logistic_rate_and_offset(
                t_scaled,
                trend.values.flatten(),
                capacities=self.capacity_prior_loc * self.y_scale,
            )

            linear_global_rate = linear_global_rate[0]
            timeoffset = timeoffset[0]
            changepoints_loc.at[0].set(linear_global_rate)

            changepoint_coefficients_distribution = dist.Laplace(
                changepoints_loc,
                jnp.ones(len(self._changepoint_t)) * self.changepoint_prior_scale,
            )

            distributions["changepoint_coefficients"] = (
                changepoint_coefficients_distribution
            )
            distributions["offset"] = dist.Normal(timeoffset, jnp.log(self.y_scale))

            distributions["capacity"] = dist.Normal(
                self.capacity_prior_loc,
                self.capacity_prior_scale,
            )

        if self._has_exogenous:
            (
                exogenous_name_distribution_pairs,
                exogenous_permutation_matrix,
            ) = get_exogenous_priors(
                X, self.exogenous_priors, self.default_exogenous_prior
            )

            distributions["exogenous_coefficients"] = OrderedDict(
                exogenous_name_distribution_pairs
            )
            extra_inputs["exogenous_permutation_matrix"] = exogenous_permutation_matrix
        else:
            extra_inputs["exogenous_permutation_matrix"] = None

        distributions["std_observation"] = dist.HalfNormal(self.noise_scale)
        extra_inputs["distributions"] = distributions

        return extra_inputs

    def _set_time_and_y_scales(self, y: pd.DataFrame):
        """
        Sets the scales for the time series data.

        Args:
            y (pd.DataFrame): Time series data.
        """

        # Set time scale
        t_days = convert_index_to_days_since_epoch(y.index)

        self.t_scale = (t_days[1:] - t_days[:-1]).mean()
        self.t_start = t_days.min() / self.t_scale

        # Set y scale
        self.y_scale = np.max(np.abs(y.values.flatten()))

    def _replace_hyperparam_nones_with_defaults(self, y):
        """
        Replaces None values in hyperparameters with default values.

        Args:
            y (pd.DataFrame): Time series data.
        """
        if self.exogenous_priors is None:
            self.exogenous_priors = {}

        if self.capacity_prior_loc is None:
            self.capacity_prior_loc = y.values.max() * 1.1
        if self.capacity_prior_scale is None:
            self.capacity_prior_scale = (y.values.max() - y.values.min()) * 0.2

    def _index_to_scaled_timearray(self, idx):
        """
        Scales the index values.

        Args:
            idx (pd.Index): Pandas Index object.

        Returns:
            np.ndarray: Scaled index values.
        """
        t_days = convert_index_to_days_since_epoch(idx)
        return (t_days) / self.t_scale - self.t_start

    def _convert_periodindex_to_floatarray(self, period_index):
        """
        Converts a pandas PeriodIndex object to a float array.

        Args:
            period_index (pd.PeriodIndex): Pandas PeriodIndex object.

        Returns:
            jnp.ndarray: Float array.
        """
        return jnp.array(self._index_to_scaled_timearray(period_index))

    def _set_changepoints_t(self, y):
        """
        Sets the array of changepoint times.

        Args:
            t (jnp.ndarray): Array of time values.
        """

        t_scaled = self._index_to_scaled_timearray(y.index)

        self._changepoint_t = get_changepoint_timeindexes(
            t=t_scaled,
            changepoint_interval=self.changepoint_interval,
            changepoint_range=self.changepoint_range,
        )

    def _get_changepoint_matrix(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Generates the changepoint coefficient matrix.

        Args:
            t (jnp.ndarray): Array of time values.

        Returns:
            jnp.ndarray: Changepoint coefficient matrix.
        """

        return get_changepoint_matrix(t, self._changepoint_t)

    def _predict_samples(self, X: pd.DataFrame, fh: ForecastingHorizon) -> pd.DataFrame:
        """
        Generates predictive samples.

        Args:
            X (pd.DataFrame): Exogenous variables.
            fh (ForecastingHorizon): Forecasting horizon.

        Returns:
            dict: Dictionary of predictive samples.
        """
        fh_dates = self.fh_to_index(fh)
        fh_as_index = pd.Index(list(fh_dates.to_numpy()))

        t = self._index_to_scaled_timearray(fh_as_index)
        changepoint_matrix = self._get_changepoint_matrix(t)

        predictive = Predictive(
            self.model, self.posterior_samples_, return_sites=["obs", *self.site_names]
        )

        if X is None and self.has_seasonality:
            X = pd.DataFrame(index=fh_as_index)

        if self.has_seasonality:
            X = self.add_seasonalities(X)

        X_array = (
            X.loc[fh_as_index].values if self.has_exogenous_or_seasonality else None
        )

        self.samples_predictive_ = predictive(
            self.rng_key,
            y=None,
            X=X_array,
            t=t,
            changepoint_matrix=changepoint_matrix,
            **self.extra_inputs_
        )
        return self.samples_predictive_
