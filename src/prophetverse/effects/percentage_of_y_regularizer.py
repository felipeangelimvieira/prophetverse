from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array


class PercentageOfYRegularizer(BaseEffect):
    """
    Regularize a wrapped effect so that
    effect_name(t) ≈ proportion(t) * y(t),
    with proportion(t) ~ Beta(α, β), α,β derived from mean & std.

    Parameters
    ----------
    effect_name : str
        The key in `predicted_effects` to regularize.
    prior_mean : float
        Target mean proportion (0 < prior_mean < 1).
    prior_std : float
        Target standard deviation of proportion (prior_std > 0).
    """

    _tags = {
        "requires_X": False,
        "applies_to": "y",  # transform() will receive the y‐DataFrame
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "requires_fit_before_transform": False,
    }

    def __init__(
        self,
        effect_name: str,
        prior_mean: float = 0.5,
        prior_std: float = 0.1,
    ):
        assert 0.0 < prior_mean < 1.0, "prior_mean must be in (0,1)"
        assert prior_std > 0.0, "prior_std must be > 0"
        self.effect_name = effect_name
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        super().__init__()

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict[str, Any]:
        """
        Convert the target `y` (passed in as X because applies_to='y')
        into a JAX array, for use in `_predict`.
        """
        y_array = series_to_tensor_or_array(X)
        return {"y": y_array}

    def _predict(
        self,
        data: Dict[str, jnp.ndarray],
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Enforce (pred_effect / y) ~ Beta(α, β) once per time‐step.
        """

        y = data["y"]
        x = predicted_effects[self.effect_name]

        # compute Beta shapes from prior mean & sd
        mu = self.prior_mean
        var = self.prior_std**2
        kappa = (mu * (1 - mu)) / var - 1.0
        alpha = mu * kappa
        beta = (1 - mu) * kappa

        # observed ratio; mask any missing y
        ratio = x / y
        mask = ~jnp.isnan(y)

        # add the Beta‐likelihood
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample(
                f"{self.effect_name}_pct_obs:ignore",
                dist.Beta(alpha, beta),
                obs=ratio,
            )

        # return the original effect so downstream modeling is unchanged
        return jnp.zeros_like(x)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "effect_name": "trend",
                "prior_mean": 0.5,
                "prior_std": 0.1,
            }
        ]
