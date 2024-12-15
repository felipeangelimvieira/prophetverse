import jax.numpy as jnp
import pandas as pd
import pytest

from prophetverse.effects.base import BaseAdditiveOrMultiplicativeEffect, BaseEffect


class ConcreteEffect(BaseAdditiveOrMultiplicativeEffect):
    """Most simple class to test abstracteffect methods."""

    _tags = {"skip_predict_if_no_match": False}

    def _predict(self, data, predicted_effects, params) -> jnp.ndarray:
        """Calculate simple effect."""
        return jnp.mean(data, axis=1, keepdims=True)


@pytest.fixture(name="effect_with_regex")
def effect_with_regex():
    """Most simple class of abstracteffect with optional regex."""
    return ConcreteEffect()


@pytest.fixture
def effect_without_regex():
    """Most simple class of abstracteffect without optional regex."""
    return ConcreteEffect()


@pytest.mark.smoke
def test__predict(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape((-1, 2))
    result = effect_with_regex.predict(data, predicted_effects={"trend": trend})
    expected_result = jnp.mean(data, axis=1).reshape((-1, 1))
    assert jnp.allclose(result, expected_result)

    call_result = effect_with_regex(data=data, predicted_effects={"trend": trend})

    assert jnp.all(call_result == result)


def test_bad_effect_mode():
    with pytest.raises(ValueError):
        BaseAdditiveOrMultiplicativeEffect(effect_mode="bad_mode")


def test_not_fitted():

    class EffectNotFitted(BaseEffect):
        _tags = {"requires_fit_before_transform": False}

    EffectNotFitted().transform(pd.DataFrame(), fh=pd.Index([]))

    class EffectMustFit(BaseEffect):
        _tags = {"requires_fit_before_transform": True}

    with pytest.raises(ValueError):
        EffectMustFit().transform(pd.DataFrame(), fh=pd.Index([]))
