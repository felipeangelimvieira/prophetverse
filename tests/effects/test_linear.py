import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.effect_apply import additive_effect, multiplicative_effect
from prophetverse.effects.linear import LinearEffect


@pytest.fixture
def linear_effect_multiplicative():
    return LinearEffect(
        id="test_linear_effect", prior=dist.Delta(1.0), effect_mode="multiplicative"
    )


@pytest.fixture
def linear_effect_additive():
    return LinearEffect(
        id="test_linear_effect", prior=dist.Delta(1.0), effect_mode="additive"
    )


def test_initialization_defaults():
    linear_effect = LinearEffect(id="test_linear_effect")
    assert isinstance(linear_effect.prior, dist.Normal)
    assert linear_effect.prior.loc == 0
    assert linear_effect.prior.scale == 0.1
    assert linear_effect.effect_mode == "multiplicative"


def test_compute_effect_multiplicative(linear_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    with seed(numpyro.handlers.seed, 0):
        result = linear_effect_multiplicative.compute_effect(trend, data)

    expected_result = multiplicative_effect(trend, data, jnp.array([1.0, 1.0]))

    assert jnp.allclose(result, expected_result)


def test_compute_effect_additive(linear_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    with seed(numpyro.handlers.seed, 0):
        result = linear_effect_additive.compute_effect(trend, data)

    expected_result = additive_effect(data, jnp.array([1.0, 1.0]))

    assert jnp.allclose(result, expected_result)
