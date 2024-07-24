import jax.numpy as jnp
import pandas as pd
import pytest

from prophetverse.effects.base import BaseAdditiveOrMultiplicativeEffect, BaseEffect


@pytest.mark.smoke
def test__predict(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape((-1, 2))
    result = effect_with_regex.predict(data, predicted_effects={"trend": trend})
    expected_result = jnp.mean(data, axis=1).reshape((-1, 1))
    assert jnp.allclose(result, expected_result)


@pytest.mark.smoke
def test_call(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape((-1, 2))
    result = effect_with_regex(data=data, predicted_effects={"trend": trend})
    expected_result = jnp.mean(data, axis=1).reshape((-1, 1))
    assert jnp.allclose(result, expected_result)


def test_bad_effect_mode():
    with pytest.raises(ValueError):
        BaseAdditiveOrMultiplicativeEffect(effect_mode="bad_mode")


def test_not_fitted():
    with pytest.raises(ValueError):
        BaseEffect().transform(pd.DataFrame(), fh=pd.Index([]))
