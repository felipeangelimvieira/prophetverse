from typing import Any, Dict, Optional, Callable
from typing import Literal

import jax.nn
import numpyro.handlers
from jax.scipy.special import expit
import numpyro.distributions as dist
import jax.numpy as jnp

import numpyro
from prophetverse.effects.target.base import BaseTargetEffect

from prophetverse.distributions import HurdleDistribution, TruncatedDiscrete


class HurdleTargetLikelihood(BaseTargetEffect):
    """Hurdle Target Effect

    Implements a Hurdle target effect for intermittent demand modeling.

    Parameters
    ----------
    noise_scale (float): Scale of the prior of the scale in the demand distribution, only applicable for
        likelihoods parameterized by a scale parameter (e.g., Negative Binomial).
    likelihood_family (str): Family of the likelihood for the demand distribution.
    zero_proba_effects_prefix (str): Prefix for the effects modeling the zero probability.
    proba_transform (callable): Transformation function to map linear predictors to probabilities.
    demand_transform (callable): Transformation function to ensure positive demand predictions.
    """

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
        likelihood_family: Literal["poisson", "negbinomial"] = "poisson",
        noise_scale: float = 1.0,
        zero_proba_effects_prefix: str = "zero_proba__",
        proba_transform: Callable[[jnp.ndarray], jnp.ndarray] = expit,
        demand_transform: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
    ):
        self.noise_scale = noise_scale
        self.zero_proba_effects_prefix = zero_proba_effects_prefix
        self.likelihood_family = likelihood_family
        self.proba_transform = proba_transform
        self.demand_transform = demand_transform

        super().__init__()

    def _fit(self, y, X, scale=1.0):
        self.scale_ = scale

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
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
        if not predicted_effects:
            return 0.0

        mean = sum(predicted_effects.values())
        mean = self.demand_transform(mean)

        return mean
