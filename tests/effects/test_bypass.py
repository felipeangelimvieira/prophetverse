"""Tests for BypassEffect."""

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro.handlers import seed

from prophetverse.effects.bypass import BypassEffect


@pytest.fixture
def bypass_effect():
    """Create a BypassEffect instance for testing (default: no validation)."""
    return BypassEffect()


@pytest.fixture
def bypass_effect_with_validation():
    """Create a BypassEffect instance with input validation enabled."""
    return BypassEffect(validate_empty_input=True)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    index = pd.date_range("2021-01-01", periods=10, freq="D")
    y = pd.DataFrame({"y": range(10)}, index=index)
    X = pd.DataFrame({"x1": range(10), "x2": range(10, 20)}, index=index)
    X_empty = pd.DataFrame(index=index)  # Empty DataFrame
    return y, X, X_empty


def test_bypass_effect_initialization():
    """Test BypassEffect can be initialized with default parameters."""
    effect = BypassEffect()
    assert effect is not None
    assert not effect.get_tag("requires_X")  # Default: doesn't require X
    assert effect.get_tag("capability:panel")
    assert effect.get_tag("capability:multivariate_input")
    assert not effect.validate_empty_input


def test_bypass_effect_initialization_with_validation():
    """Test BypassEffect can be initialized with validation enabled."""
    effect = BypassEffect(validate_empty_input=True)
    assert effect is not None
    assert effect.get_tag("requires_X")  # With validation: requires X
    assert effect.get_tag("capability:panel")
    assert effect.get_tag("capability:multivariate_input")
    assert effect.validate_empty_input


def test_bypass_effect_fit_predict_with_data(bypass_effect, sample_data):
    """Test BypassEffect with normal data (no validation)."""
    y, X, _ = sample_data

    # Fit should work without issues
    bypass_effect.fit(y=y, X=X, scale=1.0)

    # Transform and predict
    data = bypass_effect.transform(X, fh=X.index[-3:])
    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zero (scalar)
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_bypass_effect_fit_predict_without_x(bypass_effect, sample_data):
    """Test BypassEffect with no exogenous data."""
    y, _, _ = sample_data

    # Fit with no X data
    bypass_effect.fit(y=y, X=None, scale=1.0)

    # Transform and predict with no X
    data = bypass_effect.transform(None, fh=y.index[-3:])
    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_bypass_effect_with_validation_empty_x(
    bypass_effect_with_validation, sample_data
):
    """Test BypassEffect with validation accepts empty X during fit."""
    y, _, X_empty = sample_data

    # Should work fine with empty X
    bypass_effect_with_validation.fit(y=y, X=X_empty, scale=1.0)

    # Transform and predict should work
    data = bypass_effect_with_validation.transform(X_empty, fh=y.index[-3:])
    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect_with_validation.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_bypass_effect_with_validation_none_x(
    bypass_effect_with_validation, sample_data
):
    """Test BypassEffect with validation accepts None X during fit."""
    y, _, _ = sample_data

    # Should work fine with None X
    bypass_effect_with_validation.fit(y=y, X=None, scale=1.0)

    data = bypass_effect_with_validation.transform(None, fh=y.index[-3:])
    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect_with_validation.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_bypass_effect_with_validation_rejects_non_empty_x(
    bypass_effect_with_validation, sample_data
):
    """Test BypassEffect with validation rejects non-empty X during fit."""
    y, X, _ = sample_data

    # Should raise ValueError with non-empty X
    with pytest.raises(
        ValueError,
        match="BypassEffect with validate_empty_input=True requires X to be empty",
    ):
        bypass_effect_with_validation.fit(y=y, X=X, scale=1.0)


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

    predicted_effects = {}

    with seed(rng_seed=0):
        result = bypass_effect.predict(data=data, predicted_effects=predicted_effects)

    # Should return zero
    expected = jnp.array(0.0)
    assert jnp.allclose(result, expected)


def test_bypass_effect_panel_data_with_validation_empty_x():
    """Test BypassEffect with validation and panel data (empty X)."""
    effect = BypassEffect(validate_empty_input=True)

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


def test_bypass_effect_panel_data_with_validation_rejects_non_empty_x():
    """Test BypassEffect with validation rejects non-empty panel X during fit."""
    effect = BypassEffect(validate_empty_input=True)

    # Create panel data with features
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X = pd.DataFrame({"x1": range(10)}, index=idx)

    # Should raise ValueError with non-empty panel X
    with pytest.raises(
        ValueError,
        match="BypassEffect with validate_empty_input=True requires X to be empty",
    ):
        effect.fit(y=y, X=X, scale=1.0)


def test_bypass_effect_validation_error_message_includes_columns(sample_data):
    """Test that validation error includes the problematic column names."""
    y, X, _ = sample_data
    effect = BypassEffect(validate_empty_input=True)

    with pytest.raises(ValueError) as exc_info:
        effect.fit(y=y, X=X, scale=1.0)

    error_msg = str(exc_info.value)
    assert "2 columns" in error_msg
    assert "['x1', 'x2']" in error_msg
