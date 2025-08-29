"""Autoregressive (AR(1)/Random Walk) latent effect.

This effect adds a latent Gaussian AR(1) (or random walk) component to the
additive predictor. It does not require exogenous regressors and can be used
to capture residual autocorrelation or smooth time-varying structure.

Model
-----
        x_t = phi * x_{t-1} + sigma * eps_t,   eps_t ~ Normal(0, 1)

If ``mean_reverting`` is True, ``phi`` is sampled in (0, 1) via a
sigmoid-transformed Normal prior, giving a stationary AR(1). If False, ``phi``
is fixed to 1.0, yielding a random walk (unit root) process.

Returned effect values are the latent state ``x_t`` (shape (T, 1)), which the
framework will treat as an additive component.

Parameters
----------
mean_reverting : bool, default=True
        Whether to sample an AR(1) coefficient in (0,1). If False a random walk
        (phi = 1.0) is used.
sigma_prior : numpyro Distribution, default=dist.HalfNormal(1.0)
        Prior for the innovation scale ``sigma``.
phi_prior : numpyro Distribution, optional
        Base (untransformed) prior for ``phi`` before sigmoid transform when
        ``mean_reverting`` is True. Defaults to Normal(0, 1.5).

Tags
----
requires_X = False (no exogenous data needed)
applies_to = 'y' (conceptually tied to the target timeline; X is ignored)

Example
-------
>>> from prophetverse.effects.ar import AREffect  # doctest: +SKIP
>>> ar = AREffect(mean_reverting=True)             # doctest: +SKIP

Notes
-----
Implementation uses ``RecursiveLinearTransform`` to efficiently construct the
latent trajectory in a single ``numpyro.sample`` statement.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import TransformedDistribution
from numpyro.distributions.transforms import (
    AffineTransform,
    RecursiveLinearTransform,
    SigmoidTransform,
)

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.pandas import _build_full_index

__all__ = ["AREffect"]


class AREffect(BaseEffect):
    """Latent AR(1) or Random Walk effect.

    Produces a latent Gaussian time series added to the linear predictor.
    """

    _tags = {
        "capability:panel": False,
        "capability:multivariate_input": False,
        "requires_X": False,
        "applies_to": "y",
        # We still allow filtering (irrelevant here since X is None), but we
        # must know the training index -> require fit first.
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "requires_fit_before_transform": True,
    }

    def __init__(
        self,
        mean_reverting: bool = True,
        sigma_prior: Optional[dist.Distribution] = None,
        phi_prior: Optional[dist.Distribution] = None,
        innovation_prior: Optional[dist.Distribution] = None,
    ):
        self.mean_reverting = mean_reverting
        self.innovation_prior = innovation_prior
        self.sigma_prior = sigma_prior
        # Prior before sigmoid (only used if mean_reverting)
        self.phi_prior = phi_prior
        super().__init__()

        self._innovation_prior = innovation_prior
        if self._innovation_prior is None:
            self._innovation_prior = dist.Normal(0.0, 0.05)

        self._sigma_prior = sigma_prior
        if self._sigma_prior is None:
            self._sigma_prior = dist.HalfNormal(0.5)

        self._phi_prior = phi_prior
        if self._phi_prior is None:
            self._phi_prior = dist.Normal(0.0, 0.01)

    def _fit(self, y, X, scale=1.0):  # noqa: D401
        # Store training index to allow forward sampling during predict.
        self._y_index = y.index
        self.initial_value_ = jnp.array(y.iloc[0, 0]) / scale

    def _transform(self, X, fh):  # noqa: D401
        """Prepare placeholder array and positional index mapping.

        We always sample the latent path over the union of training index and the
        requested forecasting horizon so that autoregressive dependencies are
        preserved when forecasting ahead. We then remember the integer positions
        of the requested fh inside that union. `_predict` will subset to those.
        """
        train_index = self._y_index

        index_start = min(train_index.min(), fh.min())
        index_end = max(train_index.max(), fh.max())

        full_index = _build_full_index(index_start=index_start, index_end=index_end)
        full_index = full_index.sort_values()
        ix = full_index.get_indexer(fh)
        ix = jnp.array(ix, dtype=jnp.int32)
        T_full = len(full_index)
        return jnp.zeros((T_full, 1)), ix

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        if isinstance(data, tuple):
            placeholder, ix = data
        else:
            placeholder = data
            ix = jnp.arange(placeholder.shape[0], dtype=jnp.int32)

        T = placeholder.shape[0]

        sigma = numpyro.sample("sigma", self._sigma_prior)

        phi_dist = self.phi_prior
        if self.mean_reverting:
            phi_dist = TransformedDistribution(self._phi_prior, SigmoidTransform())

        phi = numpyro.sample("phi", phi_dist)
        transition_matrix = jnp.reshape(phi, (1, 1))

        eps = self._innovation_prior.expand((T, 1)).to_event(1)
        latent_full = numpyro.sample(
            "ar_state_full",
            TransformedDistribution(
                eps,
                [
                    AffineTransform(self.initial_value_, sigma),
                    RecursiveLinearTransform(
                        transition_matrix=transition_matrix,
                    ),
                ],
            ),
        )

        latent = latent_full[ix]
        return latent  # shape (len(fh),1)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [
            {"mean_reverting": True},
            {"mean_reverting": False},
        ]
