"""Tests for panel effects: GeoMichaelisMenten, GeoGeometricAdstock, GeoWeibullAdstock."""

import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.panel import (
    GeoGeometricAdstockEffect,
    GeoMichaelisMentenEffect,
    GeoWeibullAdstockEffect,
)


# ======================================================================
# GeoMichaelisMentenEffect
# ======================================================================


class TestGeoMichaelisMentenEffect:
    """Tests for Michaelis-Menten saturation effect."""

    @pytest.fixture
    def shared_effect(self):
        return GeoMichaelisMentenEffect(
            max_effect_prior=dist.Delta(2.0),
            half_saturation_prior=dist.Delta(1.0),
            shared_max_effect=True,
            shared_half_saturation=True,
        )

    @pytest.fixture
    def per_series_effect(self):
        return GeoMichaelisMentenEffect(
            max_effect_prior=dist.Delta(2.0),
            half_saturation_prior=dist.Delta(1.0),
            shared_max_effect=False,
            shared_half_saturation=False,
            max_effect_scale_hyperprior=dist.Delta(0.0),
            half_saturation_scale_hyperprior=dist.Delta(0.0),
        )

    def test_tags(self):
        effect = GeoMichaelisMentenEffect()
        assert effect.get_tag("capability:panel") is True
        assert effect.get_tag("feature:panel_hyperpriors") is True

    def test_shared_panel(self, shared_effect):
        """Shared params on (N, T, 1) panel tensor."""
        data = jnp.ones((3, 10, 1)) * 2.0
        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})

        # Analytic: max_effect * x / (half_sat + x) = 2*2/(1+2) = 4/3
        expected = (2.0 * jnp.clip(data, 1e-9, None)) / (1.0 + data)
        assert result.shape == (3, 10, 1)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_shared_univariate(self, shared_effect):
        """(T, 1) input should return (T, 1)."""
        data = jnp.array([[0.5], [1.0], [2.0]])  # (3, 1)
        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (3, 1)

    def test_per_series_zero_scale_matches_shared(self, per_series_effect):
        """Per-series with scale=0 collapses to the shared case."""
        data = jnp.ones((3, 10, 1)) * 2.0
        with seed(numpyro.handlers.seed, 0):
            result = per_series_effect.predict(data, predicted_effects={"trend": data})

        expected = (2.0 * jnp.clip(data, 1e-9, None)) / (1.0 + data)
        assert result.shape == (3, 10, 1)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_per_series_produces_different_values(self):
        """With non-zero scale, series should vary."""
        effect = GeoMichaelisMentenEffect(
            max_effect_prior=dist.Gamma(1, 1),
            half_saturation_prior=dist.Gamma(1, 1),
            shared_max_effect=False,
            shared_half_saturation=False,
            max_effect_scale_hyperprior=dist.HalfNormal(1),
            half_saturation_scale_hyperprior=dist.HalfNormal(1),
        )
        data = jnp.ones((5, 10, 1)) * 2.0
        with seed(numpyro.handlers.seed, 42):
            result = effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (5, 10, 1)
        series_means = result.mean(axis=1).squeeze()
        assert not jnp.allclose(
            series_means, series_means[0] * jnp.ones_like(series_means), atol=1e-6
        )

    def test_get_test_params(self):
        params_list = GeoMichaelisMentenEffect.get_test_params()
        assert len(params_list) >= 2
        for params in params_list:
            effect = GeoMichaelisMentenEffect(**params)
            assert isinstance(effect, GeoMichaelisMentenEffect)


# ======================================================================
# GeoGeometricAdstockEffect
# ======================================================================


class TestGeoGeometricAdstockEffect:
    """Tests for geometric adstock (carry-over) effect."""

    @pytest.fixture
    def shared_effect(self):
        return GeoGeometricAdstockEffect(
            decay_prior=dist.Delta(0.5),
            shared_decay=True,
            normalize=False,
        )

    @pytest.fixture
    def per_series_effect(self):
        return GeoGeometricAdstockEffect(
            decay_prior=dist.Delta(0.5),
            shared_decay=False,
            decay_scale_hyperprior=dist.Delta(0.0),
            normalize=False,
        )

    def test_tags(self):
        effect = GeoGeometricAdstockEffect()
        assert effect.get_tag("capability:panel") is True
        assert effect.get_tag("feature:panel_hyperpriors") is True

    def test_shared_panel(self, shared_effect):
        """Shared decay on (N, T, 1) data."""
        # Impulse at t=0 for 3 series
        data = jnp.zeros((3, 5, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})

        assert result.shape == (3, 5, 1)
        # t=0: 1.0, t=1: 0.5, t=2: 0.25, t=3: 0.125, t=4: 0.0625
        expected_seq = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        assert jnp.allclose(result[0, :, 0], expected_seq, atol=1e-5)
        # All series should be identical with shared decay
        assert jnp.allclose(result[0], result[1], atol=1e-5)

    def test_shared_univariate(self, shared_effect):
        """(T, 1) input should return (T, 1)."""
        data = jnp.zeros((5, 1))
        data = data.at[0, :].set(1.0)
        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (5, 1)

    def test_normalize(self):
        """When normalize=True, output is scaled by (1-decay)."""
        effect = GeoGeometricAdstockEffect(
            decay_prior=dist.Delta(0.5),
            shared_decay=True,
            normalize=True,
        )
        data = jnp.zeros((2, 5, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result = effect.predict(data, predicted_effects={"trend": data})

        # t=0: 1.0*0.5=0.5, t=1: 0.5*0.5=0.25, ...
        expected_seq = jnp.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
        assert jnp.allclose(result[0, :, 0], expected_seq, atol=1e-5)

    def test_per_series_zero_scale_matches_shared(
        self, shared_effect, per_series_effect
    ):
        """Per-series with scale=0 should match shared."""
        data = jnp.zeros((3, 5, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result_shared = shared_effect.predict(
                data, predicted_effects={"trend": data}
            )
        with seed(numpyro.handlers.seed, 0):
            result_per = per_series_effect.predict(
                data, predicted_effects={"trend": data}
            )
        assert jnp.allclose(result_shared, result_per, atol=1e-5)

    def test_per_series_produces_different_values(self):
        """With non-zero scale, series should vary."""
        effect = GeoGeometricAdstockEffect(
            decay_prior=dist.Beta(2, 2),
            shared_decay=False,
            decay_scale_hyperprior=dist.HalfNormal(0.5),
        )
        data = jnp.zeros((5, 10, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 42):
            result = effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (5, 10, 1)
        series_means = result.mean(axis=1).squeeze()
        assert not jnp.allclose(
            series_means, series_means[0] * jnp.ones_like(series_means), atol=1e-6
        )

    def test_get_test_params(self):
        params_list = GeoGeometricAdstockEffect.get_test_params()
        assert len(params_list) >= 2
        for params in params_list:
            effect = GeoGeometricAdstockEffect(**params)
            assert isinstance(effect, GeoGeometricAdstockEffect)


# ======================================================================
# GeoWeibullAdstockEffect
# ======================================================================


class TestGeoWeibullAdstockEffect:
    """Tests for Weibull adstock (carry-over) effect."""

    @pytest.fixture
    def shared_effect(self):
        return GeoWeibullAdstockEffect(
            scale_prior=dist.Delta(2.0),
            concentration_prior=dist.Delta(2.0),
            shared_scale=True,
            shared_concentration=True,
            max_lag=4,
        )

    @pytest.fixture
    def per_series_effect(self):
        return GeoWeibullAdstockEffect(
            scale_prior=dist.Delta(2.0),
            concentration_prior=dist.Delta(2.0),
            shared_scale=False,
            shared_concentration=False,
            scale_scale_hyperprior=dist.Delta(0.0),
            concentration_scale_hyperprior=dist.Delta(0.0),
            max_lag=4,
        )

    def test_tags(self):
        effect = GeoWeibullAdstockEffect()
        assert effect.get_tag("capability:panel") is True
        assert effect.get_tag("feature:panel_hyperpriors") is True

    def test_shared_panel(self, shared_effect):
        """Shared params on (N, T, 1) panel tensor."""
        data = jnp.zeros((3, 10, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})

        assert result.shape == (3, 10, 1)
        # All series should be identical with shared parameters
        assert jnp.allclose(result[0], result[1], atol=1e-5)
        assert jnp.allclose(result[1], result[2], atol=1e-5)

    def test_shared_univariate(self, shared_effect):
        """(T, 1) input should return (T, 1)."""
        data = jnp.zeros((10, 1))
        data = data.at[0, :].set(1.0)
        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (10, 1)

    def test_per_series_zero_scale_matches_shared(
        self, shared_effect, per_series_effect
    ):
        """Per-series with scale=0 should match shared."""
        data = jnp.zeros((3, 10, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result_shared = shared_effect.predict(
                data, predicted_effects={"trend": data}
            )
        with seed(numpyro.handlers.seed, 0):
            result_per = per_series_effect.predict(
                data, predicted_effects={"trend": data}
            )
        assert jnp.allclose(result_shared, result_per, atol=1e-5)

    def test_per_series_produces_different_values(self):
        """With non-zero scale, series should vary."""
        effect = GeoWeibullAdstockEffect(
            scale_prior=dist.Gamma(2, 1),
            concentration_prior=dist.Gamma(2, 1),
            shared_scale=False,
            shared_concentration=False,
            scale_scale_hyperprior=dist.HalfNormal(5),
            concentration_scale_hyperprior=dist.HalfNormal(5),
            max_lag=6,
        )
        data = jnp.zeros((10, 15, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 7):
            result = effect.predict(data, predicted_effects={"trend": data})
        assert result.shape == (10, 15, 1)
        series_means = result.mean(axis=1).squeeze()
        assert not jnp.allclose(
            series_means, series_means[0] * jnp.ones_like(series_means), atol=1e-6
        )

    def test_carryover_present(self, shared_effect):
        """Verify that adstock produces carryover beyond the impulse."""
        data = jnp.zeros((2, 10, 1))
        data = data.at[:, 0, :].set(1.0)

        with seed(numpyro.handlers.seed, 0):
            result = shared_effect.predict(data, predicted_effects={"trend": data})

        # At t>0 the effect should still be non-zero (carryover)
        assert jnp.any(result[:, 1:, :] > 1e-8), "Weibull adstock should have carryover"

    def test_get_test_params(self):
        params_list = GeoWeibullAdstockEffect.get_test_params()
        assert len(params_list) >= 2
        for params in params_list:
            effect = GeoWeibullAdstockEffect(**params)
            assert isinstance(effect, GeoWeibullAdstockEffect)


# ======================================================================
# Cross-cutting: import from top-level effects
# ======================================================================


def test_top_level_imports():
    """All panel effects importable from prophetverse.effects."""
    from prophetverse.effects import (
        GeoGeometricAdstockEffect,
        GeoMichaelisMentenEffect,
        GeoWeibullAdstockEffect,
    )

    assert GeoMichaelisMentenEffect is not None
    assert GeoGeometricAdstockEffect is not None
    assert GeoWeibullAdstockEffect is not None
