from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

import numpyro
import numpyro.distributions as dist
from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array
from prophetverse.distributions import GammaReparametrized
from prophetverse.effects.target.base import BaseTargetEffect


class MultivariateNormal(BaseTargetEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "hierarchical_prophet_compliant": True,
        "capability:panel": True,
        # If no columns are found, should
        # _predict be skipped?
        "requires_X": False,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(self, noise_scale=0.05, correlation_matrix_concentration=1):

        self.noise_scale = noise_scale
        self.correlation_matrix_concentration = correlation_matrix_concentration
        super().__init__()

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

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        params : Dict[str, jnp.ndarray]
            A dictionary containing the sampled parameters of the effect.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """

        y = data

        mean = 0
        for _, effect in predicted_effects.items():
            mean += effect

        if y is not None:
            y = y.squeeze(-1).T

        mean = numpyro.deterministic("mean", mean)

        is_single_series = mean.shape[0] == 1 or mean.ndim < 3
        if is_single_series:

            mean = mean.reshape((-1, 1))
            std_observation = numpyro.sample(
                "std_observation", dist.HalfNormal(jnp.array(self.noise_scale))
            )

            with numpyro.plate("time", mean.shape[-1], dim=-2):
                numpyro.sample("obs", dist.Normal(mean, std_observation), obs=y)

        else:

            std_observation = numpyro.sample(
                "std_observation",
                dist.HalfNormal(jnp.array([self.noise_scale] * mean.shape[0])),
            )

            correlation_matrix = numpyro.sample(
                "corr_matrix",
                dist.LKJCholesky(
                    mean.shape[0],
                    concentration=self.correlation_matrix_concentration,
                ),
            )

            cov_mat = (
                jnp.diag(std_observation)
                @ correlation_matrix
                @ jnp.diag(std_observation)
            )

            cov_mat = jnp.tile(jnp.expand_dims(cov_mat, axis=0), (mean.shape[1], 1, 1))

            with numpyro.plate("time", mean.shape[-1], dim=-2):
                numpyro.sample(
                    "obs",
                    dist.MultivariateNormal(mean.squeeze(-1).T, scale_tril=cov_mat),
                    obs=y,
                )

        return jnp.zeros_like(mean)
