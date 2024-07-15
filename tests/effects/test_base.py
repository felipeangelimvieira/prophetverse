import jax.numpy as jnp
import pytest


@pytest.mark.smoke
def test__predict(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex._predict(trend, data)
    expected_result = jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)


@pytest.mark.smoke
def test_call(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex(trend, data=data)
    expected_result = jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)
