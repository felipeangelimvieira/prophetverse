from typing import Any, Dict, Optional
from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpyro.handlers
import pandas as pd
from jax.scipy.special import expit
import numpyro.distributions as dist
from numpyro.distributions.transforms import (
    AffineTransform,
    RecursiveLinearTransform,
    SigmoidTransform,
)
import jax.numpy as jnp
import pandas as pd

import numpyro
from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array
from prophetverse.distributions import GammaReparametrized
from prophetverse.effects.target.base import BaseTargetEffect

from prophetverse.distributions import HurdleDistribution, TruncatedDiscrete
from prophetverse.sktime.base import BaseBayesianForecaster
from prophetverse.effects.target.univariate import _build_positive_smooth_clipper


class HurdleTargetLikelihood(BaseTargetEffect):
    """Hurdle Target Effect"""

    discrete_support = True

    _tags = {
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
        likelihood_family: Literal["poisson", "negbinomial"] = "poisson",
        zero_proba_effects_prefix="zero_proba__",
        proba_transform=expit,
        eps=1e-7,
    ):

        self.noise_scale = noise_scale
        self.zero_proba_effects_prefix = zero_proba_effects_prefix
        self.likelihood_family = likelihood_family
        self.eps = eps
        self.proba_transform = proba_transform
        self.link_function = _build_positive_smooth_clipper(eps)

        super().__init__()

    def _fit(self, y, X, scale=1):
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

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """

        zero_prob_effects = {
            k: v
            for k, v in predicted_effects.items()
            if k.startswith(self.zero_proba_effects_prefix)
        }
        demand_effects = {
            k: v
            for k, v in predicted_effects.items()
            if not k.startswith(self.zero_proba_effects_prefix)
        }
        demand = self._compute_mean(demand_effects) * self.scale_

        # If len is zero, sample a fixed small probability
        if len(zero_prob_effects) == 0:
            gate_proba_const = numpyro.sample(
                "zero_proba_const",
                dist.Beta(2, 20),
            )
            gate_prob = jnp.ones(demand.shape) * gate_proba_const
        else:
            gate_prob = self._compute_mean(zero_prob_effects)
            gate_prob = self.proba_transform(gate_prob)

        if self.likelihood_family == "negbinomial":
            noise_scale = numpyro.sample(
                "noise_scale", dist.HalfNormal(self.noise_scale)
            )
            dist_nonzero = dist.NegativeBinomial2(demand, noise_scale)
        elif self.likelihood_family == "poisson":
            dist_nonzero = dist.Poisson(demand)
        else:
            raise ValueError(f"Unknown family: {self.likelihood_family}!")

        truncated = TruncatedDiscrete(dist_nonzero, low=0)

        with numpyro.plate("data", len(demand), dim=-2):

            samples = numpyro.sample(
                "obs",
                HurdleDistribution(gate_prob, truncated),
                obs=data,
            )

        numpyro.deterministic("gate", gate_prob)
        numpyro.deterministic("demand", demand)
        numpyro.deterministic("mean", samples)

        return jnp.zeros_like(demand)

    def _compute_mean(self, predicted_effects: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mean = 0
        for _, effect in predicted_effects.items():
            mean += effect

        mean = self.link_function(mean) if self.link_function else mean
        return mean
