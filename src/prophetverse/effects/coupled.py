"""Coupled effects"""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect

__all__ = ["CoupledExactLikelihood"]


class CoupledExactLikelihood(BaseEffect):
    """Link two effects via a Normal likelihood"""

    _tags = {"requires_X": False, "hierarchical_prophet_compliant": False}

    def __init__(
        self,
        source_effect_name: str,
        target_effect_name: str,
        prior_scale: float,
    ):
        self.source_effect_name = source_effect_name
        self.target_effect_name = target_effect_name
        self.prior_scale = prior_scale
        assert prior_scale > 0, "prior_scale must be greater than 0"
        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        self.timeseries_scale = scale

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict[str, Any]:
        return {"data": None}

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        source = predicted_effects[self.source_effect_name]
        target = predicted_effects[self.target_effect_name]
        numpyro.sample(
            f"coupled_exact_likelihood:{self.target_effect_name}",
            dist.Normal(source, self.prior_scale),
            obs=target,
        )
        return target

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "source_effect_name": "trend",
                "target_effect_name": "trend",
                "prior_scale": 0.1,
            }
        ]
