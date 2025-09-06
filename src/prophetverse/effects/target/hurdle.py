"""Hurdle Target Effect for Intermittent Demand Modeling."""

from typing import Any, Dict, Optional, Union, List
from typing import Literal
import re


from typing import Any, Callable, Dict, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.handlers
from jax import Array
from jax.scipy.special import expit

from prophetverse.distributions import HurdleDistribution, TruncatedDiscrete
from prophetverse.effects.target.base import BaseTargetEffect


def softplus(x: Array) -> Array:
    """Softplus activation function.

    Parameters
    ----------
    x: Array
        Input array.

    Returns
    -------
    Returns the softplus of the input array.
    """
    return jax.nn.softplus(x)


class HurdleTargetLikelihood(BaseTargetEffect):
    """Hurdle Target Effect.

    Implements a Hurdle target effect for intermittent demand modeling.

    Parameters
    ----------
    noise_scale (float): Scale of the prior of the scale in the demand distribution,
        only applicable for likelihoods parameterized by a scale parameter
        (e.g., Negative Binomial).
    likelihood_family (str): Family of the likelihood for the demand distribution.
    zero_proba_effects_prefix (str): Prefix for the effects modeling the zero
        probability.
    proba_transform (callable): Transformation function to map linear predictors to
        probabilities.
    demand_transform (callable): Transformation function to ensure positive demand
        predictions.
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
        noise_scale=0.05,
        likelihood_family: Literal["poisson", "negbinomial"] = "poisson",
        gate_effect_names: Optional[Union[str, List[str]]] = ".*",
        gate_effect_only: Optional[Union[str, List[str]]] = None,
        proba_transform=lambda x: jnp.clip(x, 1e-5, 1 - 1e-5),
        eps=1e-7,
    ):
        """Hurdle likelihood with configurable gate (zero) effects.

        Parameters
        ----------
        noise_scale : float, default=0.05
            Scale parameter (used for Negative Binomial noise).
        likelihood_family : {"poisson", "negbinomial"}
            Distribution for the positive (non–zero) component.
        gate_effect_names : str | list[str] | None, default=".*"
            Regex (or list of regex) patterns identifying effect names that
            contribute to the gate probability. If None, no effects are used for the
            gate (a prior Beta constant is then sampled).
        gate_effect_only : str | list[str] | None, default=None
            If not None, only the effects matching these patterns are used for the
            gate. This can be useful to restrict the gate effects to a specific
            subset.
        proba_transform : callable, default=expit
            Transformation applied to the (unbounded) gate linear predictor to map
            it to (0,1).
        eps : float, default=1e-7
            Numerical stability for positive clipping of demand.
        """

        self.noise_scale = noise_scale
        self.likelihood_family = likelihood_family
        self.proba_transform = proba_transform
        self.demand_transform = demand_transform

        self.gate_effect_names = gate_effect_names
        self.gate_effect_only = gate_effect_only

        super().__init__()

    def _get_observable(self, demand: jnp.ndarray) -> dist.Distribution:
        if self.likelihood_family == "negbinomial":
            noise_scale = numpyro.sample(
                "noise_scale", dist.HalfNormal(self.noise_scale)
            )
            return dist.NegativeBinomial2(demand, noise_scale)

        if self.likelihood_family == "poisson":
            return dist.Poisson(demand)

        raise ValueError(f"Unknown family: {self.likelihood_family}!")

    def _fit(self, y, X, scale=1.0):
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

        # Split effects according to gate pattern
        gate_only_effects, common_effects, non_gate_effects = self._split_gate_effects(
            predicted_effects
        )

        # Demand effect selection

        gate_effects = {
            **gate_only_effects,
            **common_effects,
        }

        demand_effects = {
            **common_effects,
            **non_gate_effects,
        }

        demand = self._compute_mean(demand_effects) * self.scale_

        # Gate linear predictor
        if len(gate_effects) == 0:
            gate_proba_const = numpyro.sample(
                "zero_proba_const",
                dist.Beta(2, 20),
            )
            gate_prob = jnp.ones(demand.shape) * gate_proba_const
        else:
            gate_linear = 0
            for eff in gate_effects.values():
                gate_linear += eff
            gate_prob = self.proba_transform(gate_linear)

        base_dist = self._get_observable(demand)
        truncated = TruncatedDiscrete(base_dist, low=0)

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

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _match_any(self, patterns: Optional[Union[str, List[str]]], name: str) -> bool:
        """Return True if name matches provided pattern(s).

        If a list is provided each element is treated as a regex pattern. If a
        single string is provided it is compiled as regex. None returns False.
        """
        if patterns is None:
            return False
        if isinstance(patterns, str):
            return re.search(patterns, name) is not None
        # Assume iterable of patterns
        for p in patterns:
            if re.search(p, name):
                return True
        return False

    def _split_gate_effects(
        self, predicted_effects: Dict[str, jnp.ndarray]
    ) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """Split effects into (gate_effects, non_gate_effects)."""
        gate: Dict[str, jnp.ndarray] = {}
        common: Dict[str, jnp.ndarray] = {}
        rest: Dict[str, jnp.ndarray] = {}
        for name, value in predicted_effects.items():

            if self._match_any(self.gate_effect_only, name):
                gate[name] = value
            elif self._match_any(self.gate_effect_names, name):
                common[name] = value
            else:
                rest[name] = value
        return gate, common, rest
