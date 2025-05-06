import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.hill import HillEffect
from prophetverse.utils.algebric_operations import _exponent_safe


@pytest.fixture
def hill_effect_multiplicative():
    return HillEffect(
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        effect_mode="multiplicative",
    )


@pytest.fixture
def hill_effect_additive():
    return HillEffect(
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        effect_mode="additive",
    )


def test_initialization_defaults():
    hill_effect = HillEffect()
    assert isinstance(hill_effect._half_max_prior, dist.Gamma)
    assert isinstance(hill_effect._slope_prior, dist.HalfNormal)
    assert isinstance(hill_effect._max_effect_prior, dist.Gamma)
    assert hill_effect.effect_mode == "multiplicative"


def test__predict_multiplicative(hill_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([0.5, 1.0, 1.5]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = hill_effect_multiplicative.predict(
            data=data, predicted_effects={"trend": trend}
        )

    half_max, slope, max_effect = 0.5, 1.0, 1.5
    x = _exponent_safe(data / half_max, -slope)
    expected_effect = max_effect / (1 + x)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__predict_additive(hill_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([0.5, 1.0, 1.5]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = hill_effect_additive.predict(
            data=data, predicted_effects={"trend": trend}
        )

    half_max, slope, max_effect = 0.5, 1.0, 1.5
    x = _exponent_safe(data / half_max, -slope)
    expected_result = max_effect / (1 + x)

    assert jnp.allclose(result, expected_result)
