"""GeoMichaelisMentenEffect — Michaelis-Menten saturation for panel series."""

from typing import Dict, Optional

import jax.numpy as jnp
from numpyro.distributions import Distribution
from numpyro import distributions as dist

from prophetverse.effects.panel.base import BaseGeoEffect

__all__ = ["GeoMichaelisMentenEffect"]


class GeoMichaelisMentenEffect(BaseGeoEffect):
    """Michaelis-Menten saturation effect for panel (multi-series) data.

    The Michaelis-Menten equation models saturation:

    .. math::

        f(x) = \\frac{\\text{max\\_effect} \\cdot x}{
            \\text{half\\_saturation} + x
        }

    Each parameter — ``max_effect`` and ``half_saturation`` — can be
    independently configured to be **shared** or **per-series**.

    Parameters
    ----------
    max_effect_prior : Distribution, optional
        Prior (shared) or location hyperprior (per-series).
        Default: ``Gamma(1, 1)``.
    half_saturation_prior : Distribution, optional
        Prior for the half-saturation constant.
        Default: ``Gamma(1, 1)``.
    shared_max_effect : bool
        Share ``max_effect`` across series (default ``True``).
    shared_half_saturation : bool
        Share ``half_saturation`` across series (default ``True``).
    max_effect_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_max_effect=False``.
        Default: ``HalfNormal(1)``.
    half_saturation_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_half_saturation=False``.
        Default: ``HalfNormal(1)``.

    Examples
    --------
    >>> from prophetverse.effects.panel import GeoMichaelisMentenEffect
    >>> effect = GeoMichaelisMentenEffect(
    ...     shared_max_effect=False,
    ...     shared_half_saturation=True,
    ... )
    """

    def __init__(
        self,
        max_effect_prior: Optional[Distribution] = None,
        half_saturation_prior: Optional[Distribution] = None,
        shared_max_effect: bool = True,
        shared_half_saturation: bool = True,
        max_effect_scale_hyperprior: Optional[Distribution] = None,
        half_saturation_scale_hyperprior: Optional[Distribution] = None,
    ):
        self.max_effect_prior = max_effect_prior
        self.half_saturation_prior = half_saturation_prior
        self.shared_max_effect = shared_max_effect
        self.shared_half_saturation = shared_half_saturation
        self.max_effect_scale_hyperprior = max_effect_scale_hyperprior
        self.half_saturation_scale_hyperprior = half_saturation_scale_hyperprior

        self._max_effect_prior = self._resolve_prior(max_effect_prior, dist.Gamma(1, 1))
        self._half_saturation_prior = self._resolve_prior(
            half_saturation_prior, dist.Gamma(1, 1)
        )
        self._max_effect_scale_hyperprior = self._resolve_prior(
            max_effect_scale_hyperprior, dist.HalfNormal(1)
        )
        self._half_saturation_scale_hyperprior = self._resolve_prior(
            half_saturation_scale_hyperprior, dist.HalfNormal(1)
        )

        super().__init__()

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        data, was_2d = self._ensure_3d(data)
        n_series = data.shape[0]

        max_effect = self._sample_param(
            "max_effect",
            self._max_effect_prior,
            self._max_effect_scale_hyperprior,
            self.shared_max_effect,
            n_series,
        )
        half_saturation = self._sample_param(
            "half_saturation",
            self._half_saturation_prior,
            self._half_saturation_scale_hyperprior,
            self.shared_half_saturation,
            n_series,
        )

        data = jnp.clip(data, 1e-9, None)
        effect = (max_effect * data) / (half_saturation + data)

        return self._maybe_squeeze(effect)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameter sets for the skbase testing framework."""
        return [
            {
                "shared_max_effect": True,
                "shared_half_saturation": True,
            },
            {
                "shared_max_effect": False,
                "shared_half_saturation": False,
            },
        ]
