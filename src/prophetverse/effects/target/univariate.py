from typing import Any, Dict, Optional

import jax.numpy as jnp
import pandas as pd

import numpyro
import numpyro.distributions as dist
from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array
from prophetverse.distributions import GammaReparametrized
from prophetverse.effects.target.base import BaseTargetEffect


def _do_nothing(x):
    return x


class TargetLikelihood(BaseTargetEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "hierarchical_prophet_compliant": True,
        # If no columns are found, should
        # _predict be skipped?
        "requires_X": False,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
        # "capability:panel": True,
    }

    def __init__(
        self,
        noise_scale=0.05,
        link_function=None,
        likelihood_func=None,
    ):
        self.link_function = link_function
        self.noise_scale = noise_scale
        self.likelihood_func = likelihood_func
        super().__init__()

    def _fit(self, y, X, scale=1):
        self.likelihood_distribution_ = (
            dist.Normal if self.likelihood_func is None else self.likelihood_func
        )
        self.scale_ = scale

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

        mean = self._compute_mean(predicted_effects)
        mean = numpyro.deterministic("mean", mean)
        noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(self.noise_scale))

        if mean.ndim <= 2:
            mean = mean.reshape((-1, 1))

            with numpyro.plate("data", len(mean), dim=-2):
                numpyro.sample(
                    "obs",
                    self.likelihood_distribution_(mean, noise_scale),
                    obs=data,
                )
        else:
            with numpyro.plate("series", mean.shape[0], dim=-3):
                with numpyro.plate("data", mean.shape[1], dim=-2):
                    numpyro.sample(
                        "obs",
                        self.likelihood_distribution_(mean, noise_scale),
                        obs=data,
                    )
        return jnp.zeros_like(mean)

    def _compute_mean(self, predicted_effects: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mean = 0
        for _, effect in predicted_effects.items():
            mean += effect

        mean = self.link_function(mean) if self.link_function else mean
        return mean


class NormalTargetLikelihood(TargetLikelihood):
    def __init__(
        self,
        noise_scale=0.05,
    ):
        super().__init__(
            noise_scale, link_function=_do_nothing, likelihood_func=dist.Normal
        )


class GammaTargetLikelihood(TargetLikelihood):
    def __init__(
        self,
        noise_scale=0.05,
        epsilon=1e-5,
    ):
        self.epsilon = epsilon
        link_function = _build_positive_smooth_clipper(epsilon)
        super().__init__(
            noise_scale,
            link_function=link_function,
            likelihood_func=GammaReparametrized,
        )


class NegativeBinomialTargetLikelihood(TargetLikelihood):
    def __init__(
        self,
        noise_scale=0.05,
        epsilon=1e-5,
    ):
        self.epsilon = epsilon
        link_function = _build_positive_smooth_clipper(epsilon)
        super().__init__(
            noise_scale,
            link_function=link_function,
            likelihood_func=dist.NegativeBinomial2,
        )

    def _predict(self, data, predicted_effects, *args, **kwargs):
        y = data

        mean = self._compute_mean(predicted_effects) * self.scale_
        mean = numpyro.deterministic("mean", mean)
        noise_scale = numpyro.sample("noise_scale", dist.HalfNormal(self.noise_scale))

        with numpyro.plate("data", len(mean), dim=-2):
            numpyro.sample(
                "obs",
                dist.NegativeBinomial2(
                    mean.reshape((-1, 1)), noise_scale * self.scale_
                ),
                obs=y,
            )

        return jnp.zeros_like(mean)


def _build_positive_smooth_clipper(
    smooth_threshold: float, threshold: float = 1e-10
) -> jnp.ndarray:
    """Force the values of x to be positive.

    Applies a smooth threshold to the values of x to force positive-only outputs.
    Further clips the values of x to avoid zeros due to numerical precision.

    Parameters
    ----------
    smooth_threshold : float
        The threshold value for the exponential function.
    threshold : float, optional
        The threshold value for clipping, by default 1e-10.

    Returns
    -------
    jnp.ndarray
        The transformed array.
    """

    def _to_positive(x):
        return jnp.clip(
            jnp.where(
                x < smooth_threshold,
                jnp.exp(x - smooth_threshold) * smooth_threshold,
                x,
            ),
            threshold,
            None,
        )

    return _to_positive
