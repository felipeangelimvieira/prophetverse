import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.linear import LinearEffect


@pytest.fixture
def linear_effect_multiplicative():
    return LinearEffect(prior=dist.Delta(1.0), effect_mode="multiplicative")


@pytest.fixture
def linear_effect_additive():
    return LinearEffect(prior=dist.Delta(1.0), effect_mode="additive")


def test_initialization_defaults():
    linear_effect = LinearEffect()
    assert isinstance(linear_effect.prior, dist.Normal)
    assert linear_effect.prior.loc == 0
    assert linear_effect.prior.scale == 0.1
    assert linear_effect.effect_mode == "multiplicative"


def test__predict_multiplicative(linear_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    with seed(numpyro.handlers.seed, 0):
        result = linear_effect_multiplicative.predict(
            data=data, predicted_effects={"trend": trend}
        )

    expected_result = trend * (data @ jnp.array([1.0, 1.0]).reshape((-1, 1)))

    assert jnp.allclose(result, expected_result)


def test__predict_additive(linear_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    with seed(numpyro.handlers.seed, 0):
        result = linear_effect_additive.predict(
            data=data, predicted_effects={"trend": trend}
        )

    expected_result = data @ jnp.array([1.0, 1.0]).reshape((-1, 1))

    assert jnp.allclose(result, expected_result)
