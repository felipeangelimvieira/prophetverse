"""Definition of Linear Effect class."""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    EFFECT_APPLICATION_TYPE,
    BaseAdditiveOrMultiplicativeEffect,
)
from prophetverse.utils.frame_to_array import series_to_tensor

__all__ = ["LinearEffect"]


class LinearEffect(BaseAdditiveOrMultiplicativeEffect):
    """Represents a linear effect in a hierarchical prophet model.

    Parameters
    ----------
    prior : Distribution, optional
        A numpyro distribution to use as prior. Defaults to dist.Normal(0, 1)
    effect_mode : effects_application, optional
        Either "multiplicative" or "additive" by default "multiplicative".
    """

    _tags = {
        "hierarchical_prophet_compliant": True,
        "capability:multivariate_input": True,
    }

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        prior: Optional[Distribution] = None,
        broadcast=False,
    ):
        self.prior = prior
        self.broadcast = broadcast
        self._prior = self.prior if prior is not None else dist.Normal(0, 0.1)

        super().__init__(effect_mode=effect_mode)

        if self.broadcast:
            self.set_tags(**{"capability:multivariate_input": False})

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        n_features = data.shape[-1]

        with numpyro.plate("features_plate", n_features, dim=-1):
            coefficients = numpyro.sample("coefs", self._prior)

        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)

        return data @ coefficients


class PanelBHLinearEffect(BaseAdditiveOrMultiplicativeEffect):
    """
    Bayesian Hierarchical linear effect.

    Parameters
    ----------
    loc_hyperprior : Distribution, optional
        A numpyro distribution to use as the location hyperprior. Defaults to dist.Normal(0
        , 1).
    scale_hyperprior : Distribution, optional
        A numpyro distribution to use as the scale hyperprior. Defaults to dist.HalfNormal
        (1).
    prior_callable : Distribution, optional
        A numpyro distribution callable to use as the prior. Defaults to dist.Normal.
    effect_mode : effects_application, optional
        Either "multiplicative" or "additive" by default "multiplicative".
    """

    _tags = {
        "hierarchical_prophet_compliant": True,
        "capability:panel": True,
        "capability:multivariate_input": False,
        "feature:panel_hyperpriors": True,
    }

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        loc_hyperprior: Optional[Distribution] = None,
        scale_hyperprior: Optional[Distribution] = None,
        prior_callable: Optional[Distribution] = None,
    ):
        self.loc_hyperprior = loc_hyperprior
        self.scale_hyperprior = scale_hyperprior
        self.prior_callable = prior_callable

        super().__init__(
            effect_mode=effect_mode,
        )

        self._loc_hyperprior = (
            loc_hyperprior if loc_hyperprior is not None else dist.Normal(0, 1)
        )
        self._scale_hyperprior = (
            scale_hyperprior if scale_hyperprior is not None else dist.HalfNormal(1)
        )
        self._prior_callable = (
            prior_callable if prior_callable is not None else dist.Normal
        )

    def _transform(self, X, fh):
        """Prepare input data to be passed to numpyro model."""
        return series_to_tensor(X)

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray]
            A dictionary containing the predicted effects

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """

        loc_hyperprior = numpyro.sample(
            "loc_hyperprior",
            self._loc_hyperprior,
        )

        scale_hyperprior = numpyro.sample(
            "scale_hyperprior",
            self._scale_hyperprior,
        )

        args = (loc_hyperprior, scale_hyperprior)

        prior = self._prior_callable(*args)

        with numpyro.handlers.plate("panel_plate", data.shape[0], dim=-1):
            coefficients = numpyro.sample(
                "coefs",
                prior,
            )

        return data @ coefficients.reshape((-1, 1, 1))
