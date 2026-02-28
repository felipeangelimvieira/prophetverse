"""Tests for BypassEffect."""

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro.handlers import seed

from prophetverse.effects.ignore_input import IgnoreInput


@pytest.fixture
def ignore_input_effect():
    """Create a BypassEffect instance for testing (default: no validation)."""
    return IgnoreInput()


@pytest.fixture
def ignore_input_effect_with_validation():
    """Create a BypassEffect instance with input validation enabled."""
    return IgnoreInput(raise_error=True)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    index = pd.date_range("2021-01-01", periods=10, freq="D")
    y = pd.DataFrame({"y": range(10)}, index=index)
    X = pd.DataFrame({"x1": range(10), "x2": range(10, 20)}, index=index)
    X_empty = pd.DataFrame(index=index)  # Empty DataFrame
    return y, X, X_empty


def test_ignore_input_effect_fit_predict_with_data(ignore_input_effect, sample_data):
    """Test BypassEffect with normal data (no validation)."""
    y, X, _ = sample_data

    # Fit should work without issues
    ignore_input_effect.fit(y=y, X=X, scale=1.0)

    # Transform and predict
    data = ignore_input_effect.transform(X, fh=X.index[-3:])
    predicted_effects = {}

    with seed(rng_seed=0):
        result = ignore_input_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zero (scalar)
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_ignore_input_effect_with_validation_rejects_non_empty_x(
    ignore_input_effect_with_validation, sample_data
):
    """Test BypassEffect with validation rejects non-empty X during fit."""
    y, X, _ = sample_data

    # Should raise ValueError with non-empty X
    with pytest.raises(
        ValueError,
    ):
        ignore_input_effect_with_validation.fit(y=y, X=X, scale=1.0)


def test_ignore_input_effect_panel_data(ignore_input_effect):
    """Test BypassEffect with panel/hierarchical data."""
    # Create panel data
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X = pd.DataFrame({"x1": range(10)}, index=idx)

    ignore_input_effect.fit(y=y, X=X, scale=1.0)
    data = ignore_input_effect.transform(X, fh=idx.get_level_values(1).unique()[-2:])

    predicted_effects = {}

    with seed(rng_seed=0):
        result = ignore_input_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_ignore_input_effect_panel_data_with_validation_empty_x():
    """Test BypassEffect with validation and panel data (empty X)."""
    effect = IgnoreInput(raise_error=True)

    # Create panel data
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X_empty = pd.DataFrame(index=idx)  # Empty panel DataFrame

    effect.fit(y=y, X=X_empty, scale=1.0)
    data = effect.transform(X_empty, fh=idx.get_level_values(1).unique()[-2:])

    predicted_effects = {}

    with seed(rng_seed=0):
        result = effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_ignore_input_effect_panel_data_with_validation_rejects_non_empty_x():
    """Test BypassEffect with validation rejects non-empty panel X during fit."""
    effect = IgnoreInput(raise_error=True)

    # Create panel data with features
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X = pd.DataFrame({"x1": range(10)}, index=idx)

    # Should raise ValueError with non-empty panel X
    with pytest.raises(
        ValueError,
    ):
        effect.fit(y=y, X=X, scale=1.0)
