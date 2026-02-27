"""GeoGeometricAdstockEffect — Geometric adstock for panel series."""

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution
from numpyro import distributions as dist

from prophetverse.effects.panel.base import BaseGeoEffect

__all__ = ["GeoGeometricAdstockEffect"]


class GeoGeometricAdstockEffect(BaseGeoEffect):
    """Geometric adstock (carry-over) effect for panel data.

    Applies a geometric decay along the time axis of each series
    independently.  The ``decay`` parameter can be shared or per-series.

    For series *i* the adstock at time *t* is:

    .. math::

        a_{i,t} = x_{i,t} + \\text{decay}_i \\cdot a_{i,t-1}

    When ``normalize=True`` the output is scaled by ``(1 - decay)`` so
    that a unit impulse integrates to 1.

    Parameters
    ----------
    decay_prior : Distribution, optional
        Prior for the decay rate. Default: ``Beta(2, 2)``.
    shared_decay : bool
        Share ``decay`` across series (default ``True``).
    decay_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_decay=False``.
        Default: ``HalfNormal(0.1)``.
    normalize : bool
        If ``True`` multiply by ``(1 - decay)`` (default ``False``).

    Examples
    --------
    >>> from prophetverse.effects.panel import GeoGeometricAdstockEffect
    >>> effect = GeoGeometricAdstockEffect(shared_decay=False)
    """

    def __init__(
        self,
        decay_prior: Optional[Distribution] = None,
        shared_decay: bool = True,
        decay_scale_hyperprior: Optional[Distribution] = None,
        normalize: bool = False,
    ):
        self.decay_prior = decay_prior
        self.shared_decay = shared_decay
        self.decay_scale_hyperprior = decay_scale_hyperprior
        self.normalize = normalize

        self._decay_prior = self._resolve_prior(decay_prior, dist.Beta(2, 2))
        self._decay_scale_hyperprior = self._resolve_prior(
            decay_scale_hyperprior, dist.HalfNormal(0.1)
        )

        super().__init__()

    # ------------------------------------------------------------------

    @staticmethod
    def _geometric_adstock_1d(x_1d: jnp.ndarray, decay: float) -> jnp.ndarray:
        """Apply geometric adstock to a single 1-D series ``(T,)``."""

        def step(carry, current):
            new = current + decay * carry
            return new, new

        _, adstock = jax.lax.scan(step, jnp.zeros_like(x_1d[0]), x_1d)
        return adstock

    # ------------------------------------------------------------------

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        data, was_2d = self._ensure_3d(data)
        n_series = data.shape[0]

        decay = self._sample_param(
            "decay",
            self._decay_prior,
            self._decay_scale_hyperprior,
            self.shared_decay,
            n_series,
        )

        # Apply scan per-series.  For shared decay the scalar broadcasts
        # automatically; for per-series we index into the (N,1,1) array.
        results = []
        for i in range(n_series):
            d_i = decay if self.shared_decay else decay[i, 0, 0]
            adstock_i = self._geometric_adstock_1d(data[i, :, 0], d_i)  # (T,)
            results.append(adstock_i)

        effect = jnp.stack(results, axis=0)[..., jnp.newaxis]  # (N, T, 1)

        if self.normalize:
            # For per-series decay broadcast is already correct.
            effect = effect * (1 - decay)

        return self._maybe_squeeze(effect)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameter sets for the skbase testing framework."""
        return [
            {"shared_decay": True, "normalize": False},
            {"shared_decay": False, "normalize": True},
        ]
