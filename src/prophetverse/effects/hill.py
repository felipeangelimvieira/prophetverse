"""Definition of Hill Effect class."""

from typing import Dict, Optional, Any

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    EFFECT_APPLICATION_TYPE,
    BaseAdditiveOrMultiplicativeEffect,
)
from prophetverse.utils.algebric_operations import _exponent_safe

__all__ = ["HillEffect"]


class HillEffect(BaseAdditiveOrMultiplicativeEffect):
    """Represents a Hill effect in a time series model.

    Parameters
    ----------
    half_max_prior : Distribution, optional
        Prior distribution for the half-maximum parameter
    slope_prior : Distribution, optional
        Prior distribution for the slope parameter
    max_effect_prior : Distribution, optional
        Prior distribution for the maximum effect parameter
    effect_mode : effects_application, optional
        Mode of the effect (either "additive" or "multiplicative")
    """

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        half_max_prior: Optional[Distribution] = None,
        slope_prior: Optional[Distribution] = None,
        max_effect_prior: Optional[Distribution] = None,
        offset_slope: Optional[float] = 0.0,
        input_scale: Optional[float] = 1.0,
        base_effect_name="trend",
    ):
        self.half_max_prior = half_max_prior
        self.slope_prior = slope_prior
        self.max_effect_prior = max_effect_prior

        self._half_max_prior = (
            self.half_max_prior if half_max_prior is not None else dist.Gamma(1, 1)
        )
        self._slope_prior = (
            self.slope_prior if slope_prior is not None else dist.HalfNormal(10)
        )
        self._max_effect_prior = (
            self.max_effect_prior if max_effect_prior is not None else dist.Gamma(1, 1)
        )
        self.offset_slope = offset_slope
        self.input_scale = input_scale

        super().__init__(effect_mode=effect_mode, base_effect_name=base_effect_name)

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs
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
            An array with shape (T,1) for univariate timeseries.
        """
        half_max = numpyro.sample("half_max", self._half_max_prior) * self.input_scale
        slope = numpyro.sample("slope", self._slope_prior) + self.offset_slope
        max_effect = numpyro.sample("max_effect", self._max_effect_prior)

        data = jnp.clip(data, 1e-9, None)
        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)
        return effect
