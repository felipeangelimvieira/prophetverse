"""GeoHillEffect — Hill saturation for panel time series."""

from typing import Dict, Optional

import jax.numpy as jnp
from numpyro.distributions import Distribution
from numpyro import distributions as dist

from prophetverse.effects.panel.base import BaseGeoEffect
from prophetverse.utils.algebric_operations import _exponent_safe

__all__ = ["GeoHillEffect"]


class GeoHillEffect(BaseGeoEffect):
    """Hill saturation effect for panel (multi-series) time series.

    Each parameter — ``half_max``, ``slope``, ``max_effect`` — can be
    independently configured to be **shared** across all series or estimated
    **per-series** with hierarchical (partial-pooling) priors.

    The Hill function applied is:

    .. math::

        f(x) = \\frac{\\text{max\\_effect}}{
            1 + \\left(\\frac{x}{\\text{half\\_max}}\\right)^{-\\text{slope}}
        }

    Parameters
    ----------
    half_max_prior : Distribution, optional
        Prior (shared) or location hyperprior (per-series) for the
        half-maximum parameter. Default: ``Gamma(1, 1)``.
    slope_prior : Distribution, optional
        Prior for the slope. Default: ``HalfNormal(10)``.
    max_effect_prior : Distribution, optional
        Prior for the maximum effect. Default: ``Gamma(1, 1)``.
    shared_half_max : bool
        Share ``half_max`` across series (default ``True``).
    shared_slope : bool
        Share ``slope`` across series (default ``True``).
    shared_max_effect : bool
        Share ``max_effect`` across series (default ``True``).
    half_max_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_half_max=False``.
        Default: ``HalfNormal(1)``.
    slope_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_slope=False``.
        Default: ``HalfNormal(1)``.
    max_effect_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_max_effect=False``.
        Default: ``HalfNormal(1)``.
    offset_slope : float
        Constant added to the sampled slope. Default ``0.0``.
    input_scale : float
        Multiplicative scale applied to ``half_max``. Default ``1.0``.

    Examples
    --------
    >>> from prophetverse.effects.panel import GeoHillEffect
    >>> effect = GeoHillEffect(
    ...     shared_half_max=False,
    ...     shared_slope=True,
    ...     shared_max_effect=False,
    ... )
    """

    def __init__(
        self,
        half_max_prior: Optional[Distribution] = None,
        slope_prior: Optional[Distribution] = None,
        max_effect_prior: Optional[Distribution] = None,
        shared_half_max: bool = True,
        shared_slope: bool = True,
        shared_max_effect: bool = True,
        half_max_scale_hyperprior: Optional[Distribution] = None,
        slope_scale_hyperprior: Optional[Distribution] = None,
        max_effect_scale_hyperprior: Optional[Distribution] = None,
        offset_slope: float = 0.0,
        input_scale: float = 1.0,
    ):
        self.half_max_prior = half_max_prior
        self.slope_prior = slope_prior
        self.max_effect_prior = max_effect_prior
        self.shared_half_max = shared_half_max
        self.shared_slope = shared_slope
        self.shared_max_effect = shared_max_effect
        self.half_max_scale_hyperprior = half_max_scale_hyperprior
        self.slope_scale_hyperprior = slope_scale_hyperprior
        self.max_effect_scale_hyperprior = max_effect_scale_hyperprior
        self.offset_slope = offset_slope
        self.input_scale = input_scale

        self._half_max_prior = self._resolve_prior(half_max_prior, dist.Gamma(1, 1))
        self._slope_prior = self._resolve_prior(slope_prior, dist.HalfNormal(10))
        self._max_effect_prior = self._resolve_prior(max_effect_prior, dist.Gamma(1, 1))
        self._half_max_scale_hyperprior = self._resolve_prior(
            half_max_scale_hyperprior, dist.HalfNormal(1)
        )
        self._slope_scale_hyperprior = self._resolve_prior(
            slope_scale_hyperprior, dist.HalfNormal(1)
        )
        self._max_effect_scale_hyperprior = self._resolve_prior(
            max_effect_scale_hyperprior, dist.HalfNormal(1)
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

        half_max = (
            self._sample_param(
                "half_max",
                self._half_max_prior,
                self._half_max_scale_hyperprior,
                self.shared_half_max,
                n_series,
            )
            * self.input_scale
        )
        slope = (
            self._sample_param(
                "slope",
                self._slope_prior,
                self._slope_scale_hyperprior,
                self.shared_slope,
                n_series,
            )
            + self.offset_slope
        )
        max_effect = self._sample_param(
            "max_effect",
            self._max_effect_prior,
            self._max_effect_scale_hyperprior,
            self.shared_max_effect,
            n_series,
        )

        data = jnp.clip(data, 1e-9, None)
        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)

        return self._maybe_squeeze(effect)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameter sets for the skbase testing framework."""
        return [
            {
                "shared_half_max": True,
                "shared_slope": True,
                "shared_max_effect": True,
            },
            {
                "shared_half_max": False,
                "shared_slope": False,
                "shared_max_effect": False,
            },
        ]
