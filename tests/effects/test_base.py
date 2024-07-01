import jax.numpy as jnp
import pandas as pd
import pytest

from prophetverse.effects.base import AbstractEffect


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
def test_split_data_into_effects(effects_sample_data, effect_with_regex):
    effects = [effect_with_regex]
    split_data = AbstractEffect.split_data_into_effects(effects_sample_data, effects)
    assert list(split_data.keys()) == ["test_effect"]
    pd.testing.assert_frame_equal(
        split_data["test_effect"], effects_sample_data[["x1", "x2"]]
    )


@pytest.mark.smoke
def test_compute_effect(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex.compute_effect(trend, data)
    expected_result = trend + jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)


@pytest.mark.smoke
def test_call(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = effect_with_regex(trend, data)
    expected_result = trend + jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)
