import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.log import LogEffect


@pytest.fixture
def log_effect_multiplicative():
    return LogEffect(
        id="test_log_effect",
        scale_prior=dist.Delta(0.5),
        rate_prior=dist.Delta(2.0),
        effect_mode="multiplicative",
    )


@pytest.fixture
def log_effect_additive():
    return LogEffect(
        id="test_log_effect",
        scale_prior=dist.Delta(0.5),
        rate_prior=dist.Delta(2.0),
        effect_mode="additive",
    )


def test_initialization_defaults():
    log_effect = LogEffect(id="test_log_effect")
    assert isinstance(log_effect.scale_prior, dist.Gamma)
    assert isinstance(log_effect.rate_prior, dist.Gamma)
    assert log_effect.effect_mode == "multiplicative"


def test__apply_multiplicative(log_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([1.0, 2.0, 3.0])

    with seed(numpyro.handlers.seed, 0):
        result = log_effect_multiplicative.apply(trend, data=data)

    scale, rate = 0.5, 2.0
    expected_effect = scale * jnp.log(rate * data + 1)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__apply_additive(log_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([1.0, 2.0, 3.0])

    with seed(numpyro.handlers.seed, 0):
        result = log_effect_additive._apply(trend, data)

    scale, rate = 0.5, 2.0
    expected_result = scale * jnp.log(rate * data + 1)

    assert jnp.allclose(result, expected_result)


def test__apply_with_zero_data(log_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.0, 0.0, 0.0])

    with seed(numpyro.handlers.seed, 0):
        result = log_effect_multiplicative._apply(trend, data)

    scale, rate = 0.5, 2.0
    expected_effect = scale * jnp.log(rate * data + 1)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__apply_with_empty_data(log_effect_multiplicative):
    trend = jnp.array([])
    data = jnp.array([])

    with seed(numpyro.handlers.seed, 0):
        result = log_effect_multiplicative._apply(trend, data)

    scale, rate = 0.5, 2.0
    expected_effect = scale * jnp.log(rate * data + 1)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__apply_with_negative_data(log_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([-1.0, -2.0, -3.0])

    with pytest.raises(ValueError):
        log_effect_multiplicative._apply(trend, data)
