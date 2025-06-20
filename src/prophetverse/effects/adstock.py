"""Definition of Geometric Adstock Effect class."""

from typing import Dict

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
import pandas as pd
from prophetverse.effects.base import BaseEffect

__all__ = ["GeometricAdstockEffect"]


class GeometricAdstockEffect(BaseEffect):
    """Represents a Geometric Adstock effect in a time series model.

    Parameters
    ----------
    decay_prior : Distribution, optional
        Prior distribution for the decay parameter (controls the rate of decay).
    rase_error_if_fh_changes : bool, optional
        Whether to raise an error if the forecasting horizon changes during predict
    """

    _tags = {
        "hierarchical_prophet_compliant": False,
        "requires_X": True,
        "requires_fit_before_transform": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(
        self,
        decay_prior: dist.Distribution = None,
        raise_error_if_fh_changes: bool = False,
    ):
        self.decay_prior = decay_prior  # Default Beta distribution for decay rate.
        self.raise_error_if_fh_changes = raise_error_if_fh_changes
        super().__init__()

        self._min_date = None
        self._decay_prior = self.decay_prior
        if self._decay_prior is None:
            self._decay_prior = dist.Beta(2, 2)

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
        self._y = y.copy()

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
        jnp.ndarray
            the array with data for _predict

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

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs
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
        if isinstance(data, tuple):
            data, ix = data
        elif isinstance(data, (jnp.ndarray)):
            ix = jnp.arange(data.shape[0], dtype=jnp.int32)

        decay = numpyro.sample("decay", self._decay_prior)

        # Apply geometric adstock using jax.lax.scan for efficiency
        def adstock_step(carry, current):
            prev_adstock = carry
            new_adstock = current + decay * prev_adstock
            return new_adstock, new_adstock

        _, adstock = jax.lax.scan(
            adstock_step, init=jnp.array([0], dtype=data.dtype), xs=data.flatten()
        )
        adstock = adstock.reshape(-1, 1)
        adstock = adstock[ix]
        return adstock
