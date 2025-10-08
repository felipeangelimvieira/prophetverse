"""Definition of Michaelis-Menten Effect class."""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    EFFECT_APPLICATION_TYPE,
    BaseAdditiveOrMultiplicativeEffect,
)

__all__ = ["MichaelisMentenEffect"]


class MichaelisMentenEffect(BaseAdditiveOrMultiplicativeEffect):
    """Represents a Michaelis-Menten effect in a time series model.

    The Michaelis-Menten equation is commonly used in biochemistry to describe
    enzyme kinetics, but it's also useful for modeling saturation effects in
    time series analysis. The effect follows the equation:

    effect = (max_effect * data) / (half_saturation + data)

    Where:
    - max_effect is the maximum effect value (Vmax in biochemistry)
    - half_saturation is the value at which effect = max_effect/2 (Km in biochemistry)
    - data is the input variable (substrate concentration in biochemistry)

    Parameters
    ----------
    max_effect_prior : Distribution, optional
        Prior distribution for the maximum effect parameter
    half_saturation_prior : Distribution, optional
        Prior distribution for the half-saturation parameter
    effect_mode : effects_application, optional
        Mode of the effect (either "additive" or "multiplicative")
    """

    def __init__(
        self,
        effect_mode: EFFECT_APPLICATION_TYPE = "multiplicative",
        max_effect_prior: Optional[Distribution] = None,
        half_saturation_prior: Optional[Distribution] = None,
        base_effect_name="trend",
    ):
        self.max_effect_prior = max_effect_prior
        self.half_saturation_prior = half_saturation_prior

        self._max_effect_prior = (
            self.max_effect_prior if max_effect_prior is not None else dist.Gamma(1, 1)
        )
        self._half_saturation_prior = (
            self.half_saturation_prior
            if half_saturation_prior is not None
            else dist.Gamma(1, 1)
        )

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
        max_effect = numpyro.sample("max_effect", self._max_effect_prior)
        half_saturation = numpyro.sample("half_saturation", self._half_saturation_prior)

        # Clip data to avoid numerical issues with very small values
        data = jnp.clip(data, 1e-9, None)

        # Apply Michaelis-Menten equation: effect = (max_effect * data) / (half_saturation + data)
        effect = (max_effect * data) / (half_saturation + data)

        return effect
