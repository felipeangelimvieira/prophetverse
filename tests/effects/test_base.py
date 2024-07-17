import jax.numpy as jnp
import pandas as pd
import pytest

from prophetverse.effects.base import BaseAdditiveOrMultiplicativeEffect, BaseEffect


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


def test_bad_effect_mode():
    with pytest.raises(ValueError):
        BaseAdditiveOrMultiplicativeEffect(effect_mode="bad_mode")


def test_not_fitted():
    with pytest.raises(ValueError):
        BaseEffect().transform(pd.DataFrame())
