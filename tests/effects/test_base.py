import jax.numpy as jnp
import pandas as pd
import pytest

from prophetverse.effects.base import AbstractEffect


@pytest.mark.smoke
def test_match_columns_with_regex(sample_dataframe, effect_with_regex):
    matched_columns = effect_with_regex.match_columns(sample_dataframe.columns)
    assert list(matched_columns) == ["A", "B"]


@pytest.mark.smoke
def test_match_columns_without_regex(sample_dataframe, effect_without_regex):
    with pytest.raises(
        ValueError, match="To use this method, you must set the regex pattern"
    ):
        effect_without_regex.match_columns(sample_dataframe.columns)


@pytest.mark.smoke
def test_split_data_into_effects(sample_dataframe, effect_with_regex):
    effects = [effect_with_regex]
    split_data = AbstractEffect.split_data_into_effects(sample_dataframe, effects)
    assert list(split_data.keys()) == ["test_effect"]
    pd.testing.assert_frame_equal(
        split_data["test_effect"], sample_dataframe[["A", "B"]]
    )


@pytest.mark.smoke
def test_compute_effect(concrete_effect):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = concrete_effect.compute_effect(trend, data)
    expected_result = trend + jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)


@pytest.mark.smoke
def test_call(concrete_effect):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = concrete_effect(trend, data)
    expected_result = trend + jnp.mean(data, axis=0)
    assert jnp.allclose(result, expected_result)
