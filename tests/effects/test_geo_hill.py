"""Tests for GeoHillEffect."""

import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.geo_hill import GeoHillEffect
from prophetverse.utils.algebric_operations import _exponent_safe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_effect():
    """All parameters shared across series."""
    return GeoHillEffect(
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        shared_half_max=True,
        shared_slope=True,
        shared_max_effect=True,
    )


@pytest.fixture
def per_series_effect():
    """All parameters per-series (hierarchical)."""
    return GeoHillEffect(
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        shared_half_max=False,
        shared_slope=False,
        shared_max_effect=False,
        half_max_scale_hyperprior=dist.Delta(0.0),
        slope_scale_hyperprior=dist.Delta(0.0),
        max_effect_scale_hyperprior=dist.Delta(0.0),
    )


@pytest.fixture
def mixed_effect():
    """Some parameters shared, some per-series."""
    return GeoHillEffect(
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        shared_half_max=True,
        shared_slope=True,
        shared_max_effect=False,
        max_effect_scale_hyperprior=dist.Delta(0.0),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_hill(data, half_max=0.5, slope=1.0, max_effect=1.5):
    """Compute the expected Hill output analytically."""
    data = jnp.clip(data, 1e-9, None)
    x = _exponent_safe(data / half_max, -slope)
    return max_effect / (1 + x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tags():
    effect = GeoHillEffect()
    assert effect.get_tag("capability:panel") is True
    assert effect.get_tag("feature:panel_hyperpriors") is True


def test_shared_predict_univariate(shared_effect):
    """Shared params on (1, T, 1) univariate tensor."""
    data = jnp.array([[[0.5], [1.0], [1.5]]])  # (1, 3, 1)

    with seed(numpyro.handlers.seed, 0):
        result = shared_effect.predict(data, predicted_effects={"trend": data})

    expected = _expected_hill(data)
    assert jnp.allclose(
        result, expected, atol=1e-5
    ), f"Shared univariate mismatch: {result} vs {expected}"


def test_shared_predict_panel(shared_effect):
    """Shared params on (N, T, 1) panel tensor."""
    data = jnp.ones((3, 10, 1)) * 2.0  # 3 series, 10 timesteps

    with seed(numpyro.handlers.seed, 0):
        result = shared_effect.predict(data, predicted_effects={"trend": data})

    expected = _expected_hill(data)
    assert result.shape == (3, 10, 1)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_per_series_with_zero_scale_matches_shared(per_series_effect):
    """Per-series with scale=0 (Delta) should match shared behaviour.

    When the scale hyperprior is Delta(0), the non-centred parametrisation
    collapses to the location, so results should match the shared case.
    """
    data = jnp.ones((3, 10, 1)) * 2.0

    with seed(numpyro.handlers.seed, 0):
        result = per_series_effect.predict(data, predicted_effects={"trend": data})

    expected = _expected_hill(data)
    assert result.shape == (3, 10, 1)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_mixed_shared_per_series(mixed_effect):
    """Mixed: half_max and slope shared, max_effect per-series (scale=0)."""
    data = jnp.ones((3, 10, 1)) * 2.0

    with seed(numpyro.handlers.seed, 0):
        result = mixed_effect.predict(data, predicted_effects={"trend": data})

    expected = _expected_hill(data)
    assert result.shape == (3, 10, 1)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_output_shape_2d_input(shared_effect):
    """(T, 1) input should also work (non-panel fallback)."""
    data = jnp.array([[0.5], [1.0], [1.5]])  # (3, 1)

    with seed(numpyro.handlers.seed, 0):
        result = shared_effect.predict(data, predicted_effects={"trend": data})

    expected = _expected_hill(data)
    assert result.shape == data.shape
    assert jnp.allclose(result, expected, atol=1e-5)


def test_per_series_produces_different_values():
    """With non-zero scale, per-series params should vary across series."""
    effect = GeoHillEffect(
        half_max_prior=dist.Gamma(1, 1),
        slope_prior=dist.HalfNormal(5),
        max_effect_prior=dist.Gamma(1, 1),
        shared_half_max=False,
        shared_slope=False,
        shared_max_effect=False,
        half_max_scale_hyperprior=dist.HalfNormal(1),
        slope_scale_hyperprior=dist.HalfNormal(1),
        max_effect_scale_hyperprior=dist.HalfNormal(1),
    )
    data = jnp.ones((5, 10, 1)) * 2.0

    with seed(numpyro.handlers.seed, 42):
        result = effect.predict(data, predicted_effects={"trend": data})

    assert result.shape == (5, 10, 1)
    # Series should have different values (at least not all identical)
    series_means = result.mean(axis=1).squeeze()
    assert not jnp.allclose(
        series_means, series_means[0] * jnp.ones_like(series_means), atol=1e-6
    ), "Per-series params should produce variation across series"


def test_get_test_params():
    params_list = GeoHillEffect.get_test_params()
    assert len(params_list) >= 2
    for params in params_list:
        effect = GeoHillEffect(**params)
        assert isinstance(effect, GeoHillEffect)
