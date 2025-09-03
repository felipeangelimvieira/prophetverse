"""Tests for BypassEffect."""

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro.handlers import seed

from prophetverse.effects.bypass import BypassEffect


@pytest.fixture
def bypass_effect():
    """Create a BypassEffect instance for testing."""
    return BypassEffect()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    index = pd.date_range("2021-01-01", periods=10, freq="D")
    y = pd.DataFrame({"y": range(10)}, index=index)
    X = pd.DataFrame({"x1": range(10), "x2": range(10, 20)}, index=index)
    return y, X


def test_bypass_effect_initialization():
    """Test BypassEffect can be initialized."""
    effect = BypassEffect()
    assert effect is not None
    assert not effect.get_tag("requires_X")
    assert effect.get_tag("capability:panel")
    assert effect.get_tag("capability:multivariate_input")


def test_bypass_effect_fit_predict_with_data(bypass_effect, sample_data):
    """Test BypassEffect with normal data."""
    y, X = sample_data

    # Fit should work without issues
    bypass_effect.fit(y=y, X=X, scale=1.0)

    # Transform and predict
    data = bypass_effect.transform(X, fh=X.index[-3:])
    trend = jnp.ones((3, 1))  # Mock trend for 3 timesteps
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zeros with same shape as trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
    assert result.shape == (3, 1)


def test_bypass_effect_fit_predict_without_x(bypass_effect, sample_data):
    """Test BypassEffect with no exogenous data."""
    y, _ = sample_data

    # Fit with no X data
    bypass_effect.fit(y=y, X=None, scale=1.0)

    # Transform and predict with no X
    data = bypass_effect.transform(None, fh=y.index[-3:])
    trend = jnp.ones((3, 1))
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zeros with same shape as trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
    assert result.shape == (3, 1)


def test_bypass_effect_predict_without_trend(bypass_effect, sample_data):
    """Test BypassEffect when no trend is available."""
    y, X = sample_data

    bypass_effect.fit(y=y, X=X, scale=1.0)
    data = bypass_effect.transform(X, fh=X.index[-3:])

    # No trend in predicted_effects
    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should still return zeros
    assert jnp.all(result == 0)
    # Should have reasonable shape
    assert result.shape[1] == 1  # Always 1 column for effects


def test_bypass_effect_panel_data(bypass_effect):
    """Test BypassEffect with panel/hierarchical data."""
    # Create panel data
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X = pd.DataFrame({"x1": range(10)}, index=idx)

    bypass_effect.fit(y=y, X=X, scale=1.0)
    data = bypass_effect.transform(X, fh=idx.get_level_values(1).unique()[-2:])

    # Mock panel trend: shape (n_series, n_timesteps, 1)
    trend = jnp.ones((2, 2, 1))  # 2 series, 2 timesteps
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zeros with same shape as panel trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
    assert result.shape == (2, 2, 1)


def test_bypass_effect_empty_x(bypass_effect, sample_data):
    """Test BypassEffect with empty X DataFrame."""
    y, _ = sample_data
    X_empty = pd.DataFrame(index=y.index)  # Empty DataFrame with correct index

    bypass_effect.fit(y=y, X=X_empty, scale=1.0)
    data = bypass_effect.transform(X_empty, fh=y.index[-3:])

    trend = jnp.ones((3, 1))
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zeros
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
