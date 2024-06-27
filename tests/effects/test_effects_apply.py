import jax.numpy as jnp
import pytest

from prophetverse.effects.effect_apply import additive_effect, multiplicative_effect
from prophetverse.utils.algebric_operations import matrix_multiplication


def test_additive_effect_happy_path():
    data = jnp.array([[1, 2], [3, 4]])
    coefficients = jnp.array([1, 0.5])
    expected = matrix_multiplication(data, coefficients)
    result = additive_effect(data, coefficients)
    assert jnp.allclose(result, expected)


def test_multiplicative_effect_happy_path():
    trend = 2.0
    data = jnp.array([[1, 2], [3, 4]])
    coefficients = jnp.array([1, 0.5])
    expected = trend * matrix_multiplication(data, coefficients)
    result = multiplicative_effect(trend, data, coefficients)
    assert jnp.allclose(result, expected)


def test_additive_effect_mismatched_dimensions():
    data = jnp.array([[1, 2], [3, 4]])
    coefficients = jnp.array([1])
    with pytest.raises(ValueError):
        additive_effect(data, coefficients)


def test_multiplicative_effect_mismatched_dimensions():
    trend = 2.0
    data = jnp.array([[1, 2], [3, 4]])
    coefficients = jnp.array([1])
    with pytest.raises(ValueError):
        multiplicative_effect(trend, data, coefficients)


def test_additive_effect_empty_data():
    data = jnp.array([[]])
    coefficients = jnp.array([])
    expected = matrix_multiplication(data, coefficients)
    result = additive_effect(data, coefficients)
    assert jnp.allclose(result, expected)


def test_multiplicative_effect_empty_data():
    trend = 2.0
    data = jnp.array([[]])
    coefficients = jnp.array([])
    expected = trend * matrix_multiplication(data, coefficients)
    result = multiplicative_effect(trend, data, coefficients)
    assert jnp.allclose(result, expected)


def test_multiplicative_effect_zero_trend():
    trend = 0.0
    data = jnp.array([[1, 2], [3, 4]])
    coefficients = jnp.array([1, 0.5])
    expected = jnp.zeros((data.shape[0],))
    result = multiplicative_effect(trend, data, coefficients)
    assert jnp.allclose(result, expected)


def test_additive_effect_large_values():
    data = jnp.array([[1e10, 2e10], [3e10, 4e10]])
    coefficients = jnp.array([1e10, 0.5e10])
    expected = matrix_multiplication(data, coefficients)
    result = additive_effect(data, coefficients)
    assert jnp.allclose(result, expected)


def test_multiplicative_effect_large_values():
    trend = 1e10
    data = jnp.array([[1e10, 2e10], [3e10, 4e10]])
    coefficients = jnp.array([1e10, 0.5e10])
    expected = trend * matrix_multiplication(data, coefficients)
    result = multiplicative_effect(trend, data, coefficients)
    assert jnp.allclose(result, expected)
