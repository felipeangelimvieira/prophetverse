"""GeoWeibullAdstockEffect — Weibull adstock for panel series."""

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution
from numpyro import distributions as dist

from prophetverse.effects.panel.base import BaseGeoEffect

__all__ = ["GeoWeibullAdstockEffect"]


class GeoWeibullAdstockEffect(BaseGeoEffect):
    """Weibull adstock (carry-over) effect for panel data.

    Convolves each series with a Weibull PDF kernel, allowing more
    flexible carryover shapes than geometric adstock.

    Both ``scale`` and ``concentration`` (shape) can be shared or
    per-series.

    Parameters
    ----------
    scale_prior : Distribution, optional
        Prior for the Weibull scale. Default: ``Gamma(2, 1)``.
    concentration_prior : Distribution, optional
        Prior for the Weibull concentration (shape).
        Default: ``Gamma(2, 1)``.
    shared_scale : bool
        Share ``scale`` across series (default ``True``).
    shared_concentration : bool
        Share ``concentration`` across series (default ``True``).
    scale_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_scale=False``.
        Default: ``HalfNormal(1)``.
    concentration_scale_hyperprior : Distribution, optional
        Scale hyperprior when ``shared_concentration=False``.
        Default: ``HalfNormal(1)``.
    max_lag : int, optional
        Maximum lag for the Weibull kernel. Default ``13``.

    Examples
    --------
    >>> from prophetverse.effects.panel import GeoWeibullAdstockEffect
    >>> effect = GeoWeibullAdstockEffect(shared_scale=False, max_lag=8)
    """

    def __init__(
        self,
        scale_prior: Optional[Distribution] = None,
        concentration_prior: Optional[Distribution] = None,
        shared_scale: bool = True,
        shared_concentration: bool = True,
        scale_scale_hyperprior: Optional[Distribution] = None,
        concentration_scale_hyperprior: Optional[Distribution] = None,
        max_lag: int = 13,
    ):
        self.scale_prior = scale_prior
        self.concentration_prior = concentration_prior
        self.shared_scale = shared_scale
        self.shared_concentration = shared_concentration
        self.scale_scale_hyperprior = scale_scale_hyperprior
        self.concentration_scale_hyperprior = concentration_scale_hyperprior
        self.max_lag = max_lag

        self._scale_prior = self._resolve_prior(scale_prior, dist.Gamma(2, 1))
        self._concentration_prior = self._resolve_prior(
            concentration_prior, dist.Gamma(2, 1)
        )
        self._scale_scale_hyperprior = self._resolve_prior(
            scale_scale_hyperprior, dist.HalfNormal(1)
        )
        self._concentration_scale_hyperprior = self._resolve_prior(
            concentration_scale_hyperprior, dist.HalfNormal(1)
        )

        super().__init__()

    # ------------------------------------------------------------------

    @staticmethod
    def _weibull_adstock_1d(
        x_1d: jnp.ndarray,
        weights: jnp.ndarray,
        max_lag: int,
    ) -> jnp.ndarray:
        """Apply Weibull adstock to a single 1-D series ``(T,)``."""

        def step(history, current):
            adstock_val = jnp.dot(weights, history)
            new_history = jnp.concatenate([history[1:], current.reshape(1)])
            return new_history, adstock_val

        init_history = jnp.zeros(max_lag, dtype=x_1d.dtype)
        _, adstock = jax.lax.scan(step, init_history, x_1d)
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
        max_lag = self.max_lag

        scale = self._sample_param(
            "scale",
            self._scale_prior,
            self._scale_scale_hyperprior,
            self.shared_scale,
            n_series,
        )
        concentration = self._sample_param(
            "concentration",
            self._concentration_prior,
            self._concentration_scale_hyperprior,
            self.shared_concentration,
            n_series,
        )

        lags = jnp.arange(1, max_lag + 1, dtype=jnp.float32)

        # Helper to compute normalised Weibull weights for a single set
        # of scale / concentration values.
        def _weights(sc, conc):
            w = jnp.exp(dist.Weibull(scale=sc, concentration=conc).log_prob(lags))
            return w / jnp.sum(w)

        results = []
        for i in range(n_series):
            sc_i = scale if self.shared_scale else scale[i, 0, 0]
            conc_i = (
                concentration if self.shared_concentration else concentration[i, 0, 0]
            )
            w_i = _weights(sc_i, conc_i)
            adstock_i = self._weibull_adstock_1d(data[i, :, 0], w_i, max_lag)
            results.append(adstock_i)

        effect = jnp.stack(results, axis=0)[..., jnp.newaxis]  # (N, T, 1)

        return self._maybe_squeeze(effect)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameter sets for the skbase testing framework."""
        return [
            {
                "shared_scale": True,
                "shared_concentration": True,
                "max_lag": 4,
            },
            {
                "shared_scale": False,
                "shared_concentration": False,
                "max_lag": 4,
            },
        ]
