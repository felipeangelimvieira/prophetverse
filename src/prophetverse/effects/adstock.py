"""Definition of Geometric Adstock Effect class."""

from typing import Dict

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

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
        "supports_multivariate": False,
        "skip_predict_if_no_match": True,
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(
        self,
        decay_prior: dist.Distribution = None,
        raise_error_if_fh_changes: bool = True,
    ):
        self.decay_prior = decay_prior or dist.Beta(
            2, 2
        )  # Default Beta distribution for decay rate.
        self.raise_errror_if_fh_changes = raise_error_if_fh_changes
        super().__init__()

        self._min_date = None

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
        if self._min_date is None:
            self._min_date = X.index.min()
        else:
            if self._min_date != X.index.min() and self.raise_errror_if_fh_changes:
                raise ValueError(
                    "The X dataframe and forecat horizon"
                    "must be start at the same"
                    "date as the previous one"
                )
        return super()._transform(X, fh)

    def _sample_params(
        self, data: jnp.ndarray, predicted_effects: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample the parameters of the effect.

        Parameters
        ----------
        data : jnp.ndarray
            Data obtained from the transformed method.
        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.
        """
        return {
            "decay": numpyro.sample("decay", self.decay_prior),
        }

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
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
        decay = params["decay"]

        # Apply geometric adstock using jax.lax.scan for efficiency
        def adstock_step(carry, current):
            prev_adstock = carry
            new_adstock = current + decay * prev_adstock
            return new_adstock, new_adstock

        _, adstock = jax.lax.scan(
            adstock_step, init=jnp.array([0], dtype=data.dtype), xs=data
        )
        return adstock.reshape(-1, 1)
