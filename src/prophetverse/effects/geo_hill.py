"""Definition of GeoHillEffect class for panel time series."""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.algebric_operations import _exponent_safe
from prophetverse.utils.frame_to_array import series_to_tensor

__all__ = ["GeoHillEffect"]


class GeoHillEffect(BaseEffect):
    """Hill saturation effect for panel (multi-series) time series.

    Extends the standard Hill effect to support panel data, where each
    parameter (``half_max``, ``slope``, ``max_effect``) can be independently
    configured to be **shared** across all series or estimated
    **per-series** with optional hierarchical (partial-pooling) priors.

    When a parameter is shared (the default), a single value is sampled
    and applied identically to every series. When a parameter is per-series,
    a hyperprior (location/scale) is sampled first, then each series draws
    its own value from a distribution conditioned on the hyperprior. This
    encourages parameters to stay close while allowing series-specific
    deviations.

    Parameters
    ----------
    half_max_prior : Distribution, optional
        Prior for the half-maximum parameter. When ``shared_half_max=True``
        this is sampled once. When ``shared_half_max=False`` it is used as
        the **hyperprior for the location** of the per-series distribution.
        Default: ``Gamma(1, 1)``.
    slope_prior : Distribution, optional
        Prior for the slope parameter. Semantics follow the same shared /
        per-series logic as ``half_max_prior``.
        Default: ``HalfNormal(10)``.
    max_effect_prior : Distribution, optional
        Prior for the maximum-effect parameter. Semantics follow the same
        shared / per-series logic as ``half_max_prior``.
        Default: ``Gamma(1, 1)``.
    shared_half_max : bool, optional
        If ``True`` (default) a single ``half_max`` is shared across all
        series. If ``False``, each series gets its own ``half_max`` via
        a hierarchical prior.
    shared_slope : bool, optional
        If ``True`` (default) a single ``slope`` is shared across all
        series. If ``False``, each series gets its own ``slope`` via
        a hierarchical prior.
    shared_max_effect : bool, optional
        If ``True`` (default) a single ``max_effect`` is shared across all
        series. If ``False``, each series gets its own ``max_effect`` via
        a hierarchical prior.
    half_max_scale_hyperprior : Distribution, optional
        Scale hyperprior used when ``shared_half_max=False``.
        Default: ``HalfNormal(1)``.
    slope_scale_hyperprior : Distribution, optional
        Scale hyperprior used when ``shared_slope=False``.
        Default: ``HalfNormal(1)``.
    max_effect_scale_hyperprior : Distribution, optional
        Scale hyperprior used when ``shared_max_effect=False``.
        Default: ``HalfNormal(1)``.
    offset_slope : float, optional
        Constant offset added to the sampled slope. Default: ``0.0``.
    input_scale : float, optional
        Multiplicative scale applied to the sampled ``half_max``.
        Default: ``1.0``.

    Notes
    -----
    The Hill function applied is:

    .. math::

        f(x) = \\frac{\\text{max\\_effect}}{
            1 + \\left(\\frac{x}{\\text{half\\_max}}\\right)^{-\\text{slope}}
        }

    For panel data the input tensor has shape ``(N, T, 1)`` where *N* is
    the number of series and *T* the number of time steps. Parameters that
    are per-series are broadcast along the first axis.

    Examples
    --------
    Shared slope but per-series half_max and max_effect:

    >>> from prophetverse.effects.geo_hill import GeoHillEffect
    >>> effect = GeoHillEffect(
    ...     shared_half_max=False,
    ...     shared_slope=True,
    ...     shared_max_effect=False,
    ... )
    """

    _tags = {
        "capability:panel": True,
        "feature:panel_hyperpriors": True,
    }

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

        self._half_max_prior = (
            half_max_prior if half_max_prior is not None else dist.Gamma(1, 1)
        )
        self._slope_prior = (
            slope_prior if slope_prior is not None else dist.HalfNormal(10)
        )
        self._max_effect_prior = (
            max_effect_prior if max_effect_prior is not None else dist.Gamma(1, 1)
        )
        self._half_max_scale_hyperprior = (
            half_max_scale_hyperprior
            if half_max_scale_hyperprior is not None
            else dist.HalfNormal(1)
        )
        self._slope_scale_hyperprior = (
            slope_scale_hyperprior
            if slope_scale_hyperprior is not None
            else dist.HalfNormal(1)
        )
        self._max_effect_scale_hyperprior = (
            max_effect_scale_hyperprior
            if max_effect_scale_hyperprior is not None
            else dist.HalfNormal(1)
        )

        super().__init__()

    def _fit(self, y, X, scale=1.0):
        """Store panel metadata during fit.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series.
        X : pd.DataFrame
            Exogenous variables.
        scale : float
            Scale of the time series.
        """
        self._is_panel = y.index.nlevels > 1

    def _transform(self, X, fh):
        """Convert panel DataFrame to a JAX tensor.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous variables, possibly with a MultiIndex for panel data.
        fh : pd.Index
            Forecasting horizon.

        Returns
        -------
        jnp.ndarray
            Shape ``(N, T, 1)`` for panel or ``(1, T, 1)`` for univariate.
        """
        return series_to_tensor(X)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_param(
        self,
        name: str,
        prior: Distribution,
        scale_hyperprior: Distribution,
        shared: bool,
        n_series: int,
    ) -> jnp.ndarray:
        """Sample a scalar (shared) or per-series (hierarchical) parameter.

        Parameters
        ----------
        name : str
            Name for the numpyro sample site.
        prior : Distribution
            When ``shared=True`` the parameter is sampled directly from this
            distribution. When ``shared=False`` it serves as the hyperprior
            for the **location** of the per-series distribution.
        scale_hyperprior : Distribution
            Scale hyperprior used only when ``shared=False``.
        shared : bool
            Whether the parameter is shared across series.
        n_series : int
            Number of panel series (used when ``shared=False``).

        Returns
        -------
        jnp.ndarray
            Scalar when shared, shape ``(n_series, 1, 1)`` when per-series.
        """
        if shared:
            return numpyro.sample(name, prior)

        # Hierarchical: sample location and scale hyperpriors, then per-series
        loc = numpyro.sample(f"{name}_loc", prior)
        scale = numpyro.sample(f"{name}_scale", scale_hyperprior)

        with numpyro.plate(f"{name}_plate", n_series):
            raw = numpyro.sample(f"{name}_raw", dist.Normal(0, 1))

        # Non-centred parametrisation: param = loc + scale * raw
        param = loc + scale * raw
        # Reshape to (n_series, 1, 1) for broadcasting with (N, T, 1) data
        return param.reshape((-1, 1, 1))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Apply the Hill saturation function.

        Parameters
        ----------
        data : jnp.ndarray
            Input tensor of shape ``(N, T, 1)`` for panel data or
            ``(T, 1)`` / ``(1, T, 1)`` for univariate.
        predicted_effects : Dict[str, jnp.ndarray]
            Previously computed effects (e.g. trend).

        Returns
        -------
        jnp.ndarray
            Effect values with the same shape as *data*.
        """
        n_series = data.shape[0] if data.ndim == 3 else 1

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

        # Ensure 3D for uniform computation
        is_2d = data.ndim == 2
        if is_2d:
            data = data[jnp.newaxis, ...]

        data = jnp.clip(data, 1e-9, None)
        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)

        # Squeeze leading dim for single-series (non-panel) data
        if not getattr(self, "_is_panel", False) and effect.shape[0] == 1:
            effect = effect.squeeze(axis=0)

        return effect

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameter sets for skbase testing framework.

        Returns
        -------
        list of dict
            Each dict is a valid ``__init__`` kwarg set.
        """
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
