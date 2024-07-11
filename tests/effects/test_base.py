import jax.numpy as jnp
import pytest


@pytest.mark.smoke
def test_match_columns_with_regex(effects_sample_data, effect_with_regex):
    matched_columns = effect_with_regex.match_columns(effects_sample_data.columns)
    assert list(matched_columns) == ["x1", "x2"]


@pytest.mark.smoke
def test_match_columns_without_regex(effects_sample_data, effect_without_regex):
    with pytest.raises(
        ValueError, match="To use this method, you must set the regex pattern"
    ):
        effect_without_regex.match_columns(effects_sample_data.columns)


@pytest.mark.smoke
def test__apply(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex._apply(trend, data)
    expected_result = jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)


@pytest.mark.smoke
def test_call(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex(trend, data=data)
    expected_result = jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)
