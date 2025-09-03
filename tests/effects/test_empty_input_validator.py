"""Tests for EmptyInputValidator."""

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro.handlers import seed

from prophetverse.effects.empty_input_validator import EmptyInputValidator


@pytest.fixture
def validator_effect():
    """Create an EmptyInputValidator instance for testing."""
    return EmptyInputValidator()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    index = pd.date_range("2021-01-01", periods=10, freq="D")
    y = pd.DataFrame({"y": range(10)}, index=index)
    X = pd.DataFrame({"x1": range(10), "x2": range(10, 20)}, index=index)
    X_empty = pd.DataFrame(index=index)  # Empty DataFrame
    return y, X, X_empty


def test_validator_initialization():
    """Test EmptyInputValidator can be initialized."""
    effect = EmptyInputValidator()
    assert effect is not None
    assert effect.get_tag("requires_X")
    assert effect.get_tag("capability:panel")
    assert effect.get_tag("capability:multivariate_input")


def test_validator_fit_with_empty_x(validator_effect, sample_data):
    """Test EmptyInputValidator accepts empty X during fit."""
    y, _, X_empty = sample_data

    # Should work fine with empty X
    validator_effect.fit(y=y, X=X_empty, scale=1.0)

    # Transform and predict should work
    data = validator_effect.transform(X_empty, fh=y.index[-3:])
    trend = jnp.ones((3, 1))
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = validator_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zeros with same shape as trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
    assert result.shape == (3, 1)


def test_validator_fit_with_non_empty_x_raises_error(validator_effect, sample_data):
    """Test EmptyInputValidator raises error with non-empty X during fit."""
    y, X, _ = sample_data

    # Should raise error with non-empty X
    with pytest.raises(ValueError, match="EmptyInputValidator requires X to be empty"):
        validator_effect.fit(y=y, X=X, scale=1.0)


def test_validator_fit_with_single_column_x_raises_error(validator_effect, sample_data):
    """Test EmptyInputValidator raises error even with single column X."""
    y, X, _ = sample_data
    X_single = X[["x1"]]  # Single column

    # Should raise error even with single column
    with pytest.raises(ValueError, match="EmptyInputValidator requires X to be empty"):
        validator_effect.fit(y=y, X=X_single, scale=1.0)


def test_validator_fit_with_none_x(validator_effect, sample_data):
    """Test EmptyInputValidator works with None X during fit."""
    y, _, _ = sample_data

    # Should work fine with None X
    validator_effect.fit(y=y, X=None, scale=1.0)

    # Transform and predict should work
    data = validator_effect.transform(None, fh=y.index[-3:])
    trend = jnp.ones((3, 1))
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = validator_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zeros with same shape as trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)


def test_validator_predict_without_trend(validator_effect, sample_data):
    """Test EmptyInputValidator when no trend is available."""
    y, _, X_empty = sample_data

    validator_effect.fit(y=y, X=X_empty, scale=1.0)
    data = validator_effect.transform(X_empty, fh=y.index[-3:])

    # No trend in predicted_effects
    predicted_effects = {}

    with seed(rng_seed=0):
        result = validator_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should still return zeros
    assert jnp.all(result == 0)
    # Should have reasonable shape
    assert result.shape[1] == 1  # Always 1 column for effects


def test_validator_panel_data(validator_effect):
    """Test EmptyInputValidator with panel/hierarchical data."""
    # Create panel data
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X_empty = pd.DataFrame(index=idx)  # Empty panel DataFrame

    validator_effect.fit(y=y, X=X_empty, scale=1.0)
    data = validator_effect.transform(X_empty, fh=idx.get_level_values(1).unique()[-2:])

    # Mock panel trend: shape (n_series, n_timesteps, 1)
    trend = jnp.ones((2, 2, 1))  # 2 series, 2 timesteps
    predicted_effects = {"trend": trend}

    with seed(rng_seed=0):
        result = validator_effect.predict(
            data=data, predicted_effects=predicted_effects
        )

    # Should return zeros with same shape as panel trend
    expected = jnp.zeros_like(trend)
    assert jnp.allclose(result, expected)
    assert result.shape == (2, 2, 1)


def test_validator_panel_data_with_non_empty_x_raises_error(validator_effect):
    """Test EmptyInputValidator raises error with non-empty panel X."""
    # Create panel data with columns
    idx = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2021-01-01", periods=5)]
    )
    y = pd.DataFrame({"y": range(10)}, index=idx)
    X = pd.DataFrame({"x1": range(10)}, index=idx)  # Non-empty panel DataFrame

    # Should raise error
    with pytest.raises(ValueError, match="EmptyInputValidator requires X to be empty"):
        validator_effect.fit(y=y, X=X, scale=1.0)


def test_validator_error_message_includes_column_info(validator_effect, sample_data):
    """Test error message includes information about the columns found."""
    y, X, _ = sample_data

    # Should raise error with column information
    with pytest.raises(ValueError) as exc_info:
        validator_effect.fit(y=y, X=X, scale=1.0)

    error_msg = str(exc_info.value)
    assert "2 columns" in error_msg
    assert "['x1', 'x2']" in error_msg
