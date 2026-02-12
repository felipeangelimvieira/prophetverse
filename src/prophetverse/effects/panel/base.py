"""Base class for panel (geo-level) effects with hierarchical hyperpriors."""

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor

__all__ = ["BaseGeoEffect"]


class BaseGeoEffect(BaseEffect):
    """Abstract base for panel effects with shared / per-series parameters.

    Subclasses only need to implement :meth:`_predict` (the numpyro model).
    This base provides:

    * ``_fit``  – stores ``_is_panel`` flag.
    * ``_transform`` – converts a DataFrame to an ``(N, T, C)`` tensor via
      :func:`series_to_tensor`.
    * ``_sample_param`` – samples a parameter either as a single scalar
      (shared) or per-series using a non-centred hierarchical
      parametrisation.
    * ``_ensure_3d`` / ``_maybe_squeeze`` – helpers to normalise input
      shape and restore 2-D output for single-series data.

    Tags
    ----
    capability:panel : True
    feature:panel_hyperpriors : True
    """

    _tags = {
        "capability:panel": True,
        "feature:panel_hyperpriors": True,
    }

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

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
        """Convert a panel DataFrame to a JAX tensor.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous variables.
        fh : pd.Index
            Forecasting horizon.

        Returns
        -------
        jnp.ndarray
            Shape ``(N, T, C)`` for panel or ``(1, T, C)`` for univariate.
        """
        return series_to_tensor(X)

    # ------------------------------------------------------------------
    # Hierarchical sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_param(
        name: str,
        prior: Distribution,
        scale_hyperprior: Distribution,
        shared: bool,
        n_series: int,
    ) -> jnp.ndarray:
        """Sample a scalar (shared) or per-series (hierarchical) parameter.

        When ``shared=True`` the parameter is sampled once from *prior*.
        When ``shared=False`` a non-centred hierarchical parametrisation is
        used::

            loc   ~ prior
            scale ~ scale_hyperprior
            raw_i ~ Normal(0, 1)         for i in 1..n_series
            param_i = loc + scale * raw_i

        Parameters
        ----------
        name : str
            Name for the numpyro sample site.
        prior : Distribution
            Prior (shared) or location hyperprior (per-series).
        scale_hyperprior : Distribution
            Scale hyperprior (only used when ``shared=False``).
        shared : bool
            Whether the parameter is shared across all series.
        n_series : int
            Number of panel series.

        Returns
        -------
        jnp.ndarray
            Scalar when shared; shape ``(n_series, 1, 1)`` when per-series.
        """
        if shared:
            return numpyro.sample(name, prior)

        loc = numpyro.sample(f"{name}_loc", prior)
        scale = numpyro.sample(f"{name}_scale", scale_hyperprior)

        with numpyro.plate(f"{name}_plate", n_series):
            raw = numpyro.sample(f"{name}_raw", dist.Normal(0, 1))

        param = loc + scale * raw
        return param.reshape((-1, 1, 1))

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_3d(data: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        """Promote 2-D data ``(T, C)`` to ``(1, T, C)``.

        Returns the (possibly promoted) array and a flag indicating whether
        promotion happened.
        """
        if data.ndim == 2:
            return data[jnp.newaxis, ...], True
        return data, False

    def _maybe_squeeze(self, effect: jnp.ndarray) -> jnp.ndarray:
        """Squeeze the leading axis when running on single-series data.

        The framework test harness expects ``(T, C)`` for univariate and
        ``(N, T, C)`` for panel.  ``_is_panel`` is set during ``_fit``.
        """
        if not getattr(self, "_is_panel", False) and effect.shape[0] == 1:
            return effect.squeeze(axis=0)
        return effect

    # ------------------------------------------------------------------
    # Convenience: resolve prior with fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_prior(
        user_prior: Optional[Distribution],
        default: Distribution,
    ) -> Distribution:
        """Return *user_prior* if not ``None``, else *default*."""
        return user_prior if user_prior is not None else default
