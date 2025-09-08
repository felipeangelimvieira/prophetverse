"""Definition of Adstock Effect classes."""

from typing import Dict

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
import pandas as pd
from prophetverse.effects.base import BaseEffect

__all__ = ["GeometricAdstockEffect", "WeibullAdstockEffect"]


class BaseAdstockEffect(BaseEffect):
    """Base class for adstock effects.

    Contains shared functionality for handling historical data concatenation
    and common preprocessing steps for adstock computations.

    Parameters
    ----------
    raise_error_if_fh_changes : bool, optional
        Whether to raise an error if the forecasting horizon changes during predict.
    """

    _tags = {
        "hierarchical_prophet_compliant": False,
        "requires_X": True,
        "requires_fit_before_transform": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(self, raise_error_if_fh_changes: bool = False):
        self.raise_error_if_fh_changes = raise_error_if_fh_changes
        super().__init__()
        self._min_date = None

    def _fit(self, y, X, scale=1):
        """Fit the effect to the data.

        Parameters
        ----------
        y : jnp.ndarray
            The target variable.
        X : jnp.ndarray
            The exogenous variables.
        scale : float, optional
            Scale factor for the effect (default is 1).
        """
        self._X = X.copy()

    def _transform(self, X, fh):
        """Transform the dataframe and horizon to array.

        Parameters
        ----------
        X : pd.DataFrame
            dataframe with exogenous variables
        fh : pd.Index
            Forecast horizon

        Returns
        -------
        tuple
            (data, ix) where data is the transformed array and ix are the indices

        Raises
        ------
        ValueError
            If the forecasting horizon is different during predict and fit.
        """

        _X = X
        dates_to_add = self._X.index.difference(X.index)
        if len(dates_to_add) > 0:
            if self.raise_error_if_fh_changes:
                raise ValueError(
                    "The X dataframe and forecast horizon"
                    "must be start at the same"
                    "date as the previous one"
                )
            _X = pd.concat([self._X.loc[dates_to_add, X.columns], X], axis=0)
            _X = _X.sort_index()

        # Get integer location of indexes of X in _X
        ix = _X.index.get_indexer(X.index)
        ix = jnp.array(ix, dtype=jnp.int32)

        data = super()._transform(_X, fh)
        return data, ix

    def _extract_data_and_indices(self, data):
        """Extract data array and indices from the transform output.

        Parameters
        ----------
        data : jnp.ndarray or tuple
            Data obtained from the transformed method.

        Returns
        -------
        tuple
            (data_array, indices) where data_array is the transformed data
            and indices are the selection indices.
        """
        if isinstance(data, tuple):
            data_array, ix = data
        elif isinstance(data, (jnp.ndarray)):
            data_array = data
            ix = jnp.arange(data_array.shape[0], dtype=jnp.int32)
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")

        return data_array, ix


class GeometricAdstockEffect(BaseAdstockEffect):
    """Represents a Geometric Adstock effect in a time series model.

    Parameters
    ----------
    decay_prior : Distribution, optional
        Prior distribution for the decay parameter (controls the rate of decay).
    raise_error_if_fh_changes : bool, optional
        Whether to raise an error if the forecasting horizon changes during predict
    normalize : bool, optional, default False
        If True, normalizes the adstock so that the cumulative effect of a
        single-period impulse equals 1. Technically this multiplies the raw
        geometric carryover series by ``(1 - decay)`` which produces weights
        ``(1 - decay) * decay**k`` (k >= 0) that sum to 1. When False (the
        historical default behaviour), an impulse has total mass
        ``1 / (1 - decay)`` and the absolute scale of the adstock increases as
        ``decay`` approaches 1. Keeping the default as False preserves
        backwards compatibility with earlier versions.
    """

    def __init__(
        self,
        decay_prior: dist.Distribution = None,
        raise_error_if_fh_changes: bool = False,
        normalize: bool = False,
    ):
        self.decay_prior = decay_prior  # Default Beta distribution for decay rate.
        self.normalize = normalize
        super().__init__(raise_error_if_fh_changes=raise_error_if_fh_changes)

        self._decay_prior = self.decay_prior
        if self._decay_prior is None:
            self._decay_prior = dist.Beta(2, 2)

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Apply and return the geometric adstock effect values.

        Parameters
        ----------
        data : jnp.ndarray
            Data obtained from the transformed method (shape: T, 1).
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.
        params : Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.

        Returns
        -------
        jnp.ndarray
            An array with shape (T, 1) for univariate timeseries.
        """
        data_array, ix = self._extract_data_and_indices(data)

        decay = numpyro.sample("decay", self._decay_prior)

        # Apply geometric adstock using jax.lax.scan for efficiency
        def adstock_step(carry, current):
            prev_adstock = carry
            new_adstock = current + decay * prev_adstock
            return new_adstock, new_adstock

        _, adstock = jax.lax.scan(
            adstock_step,
            init=jnp.array([0], dtype=data_array.dtype),
            xs=data_array.flatten(),
        )
        adstock = adstock.reshape(-1, 1)
        # Optional normalization so that a unit impulse integrates to 1 over time
        if self.normalize:
            adstock = adstock * (1 - decay)
        adstock = adstock[ix]
        return adstock

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"normalize": True}, {"normalize": False}]


class WeibullAdstockEffect(BaseAdstockEffect):
    """Represents a Weibull Adstock effect in a time series model.

    The Weibull adstock applies a convolution of the input with a Weibull
    probability density function, allowing for more flexible carryover patterns
    compared to geometric adstock.

    Parameters
    ----------
    scale_prior : Distribution, optional
        Prior distribution for the scale parameter of the Weibull distribution.
        If None, defaults to GammaReparametrized(2, 1).
    concentration_prior : Distribution, optional
        Prior distribution for the concentration (shape) parameter of the Weibull distribution.
        If None, defaults to GammaReparametrized(2, 1).
    max_lag : int, optional
        Maximum lag to consider for the adstock effect. If None, automatically
        determined based on the Weibull distribution parameters.
    """

    def __init__(
        self,
        scale_prior: dist.Distribution = None,
        concentration_prior: dist.Distribution = None,
        max_lag: int = None,
        initial_history: float = None,
    ):
        from prophetverse.distributions import GammaReparametrized

        self.scale_prior = scale_prior
        self.concentration_prior = concentration_prior
        self.max_lag = max_lag
        self.initial_history = initial_history
        super().__init__(raise_error_if_fh_changes=False)

        self._scale_prior = self.scale_prior
        if self._scale_prior is None:
            self._scale_prior = GammaReparametrized(2, 1)

        self._concentration_prior = self.concentration_prior
        if self._concentration_prior is None:
            self._concentration_prior = GammaReparametrized(2, 1)

    def _fit(self, y, X, scale=1):

        # Determine max_lag if not provided
        self.max_lag_ = self.max_lag
        if self.max_lag_ is None:
            # Use a heuristic based on Weibull parameters
            # For Weibull distribution, most of the mass is within scale * concentration^(1/concentration) * 3
            # This is a rough approximation for the 99th percentile
            max_lag = (
                jnp.ceil(
                    scale
                    * (
                        self._concentration_prior.mean
                        ** (1.0 / self._concentration_prior.mean)
                    )
                    * 3
                )
            ).astype(int) + 1
            self.max_lag_ = jnp.clip(max_lag, 1, len(X))
        else:
            self.max_lag_ = min(self.max_lag_, len(X))

        self.lags_ = jnp.arange(1, self.max_lag_ + 1, dtype=jnp.float32)

        return super()._fit(y, X, scale)

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Apply and return the Weibull adstock effect values.

        Parameters
        ----------
        data : jnp.ndarray or tuple
            Data obtained from the transformed method (shape: T, 1) or tuple (data, ix).
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        jnp.ndarray
            An array with shape (T, 1) for univariate timeseries.
        """
        data_array, ix = self._extract_data_and_indices(data)

        # Sample Weibull parameters
        scale = numpyro.sample("scale", self._scale_prior)
        concentration = numpyro.sample("concentration", self._concentration_prior)

        # Create lag indices
        lags = self.lags_

        # Compute Weibull PDF weights for each lag
        weibull_dist = dist.Weibull(scale=scale, concentration=concentration)
        weights = jnp.exp(weibull_dist.log_prob(lags))

        # Normalize weights so they sum to 1
        weights = weights / jnp.sum(weights)

        # Apply Weibull adstock using convolution
        def weibull_adstock_step(carry, current_input):
            # carry contains the history buffer [x_{t-max_lag+1}, ..., x_{t-1}]
            history_buffer = carry

            # Compute adstock as weighted sum of current input and history
            # adstock[t] = weights[0] * x[t] (if we include current period)
            # For traditional adstock, we usually exclude current period:
            # adstock[t] = sum_{i=1}^{max_lag} weights[i-1] * x[t-i]
            adstock_value = jnp.dot(weights, history_buffer)

            # Update history buffer: shift left and add current input
            new_history = jnp.concatenate(
                [history_buffer[1:], current_input.reshape(1)]
            )

            return new_history, adstock_value

        # Initialize history buffer with zeros
        initial_history = jnp.ones(self.max_lag_, dtype=data_array.dtype)

        if self.initial_history is not None:
            initial_history *= self.initial_history
        else:
            initial_history *= data_array.flatten()[0]

        # Apply scan over the data
        _, adstock = jax.lax.scan(
            weibull_adstock_step, init=initial_history, xs=data_array.flatten()
        )

        adstock = adstock.reshape(-1, 1)
        adstock = adstock[ix]
        return adstock

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"max_lag": 4, "initial_history": 0.0}, {}]
