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
        id="test_hill_effect",
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        effect_mode="multiplicative",
    )


@pytest.fixture
def hill_effect_additive():
    return HillEffect(
        id="test_hill_effect",
        half_max_prior=dist.Delta(0.5),
        slope_prior=dist.Delta(1.0),
        max_effect_prior=dist.Delta(1.5),
        effect_mode="additive",
    )


def test_initialization_defaults():
    hill_effect = HillEffect(id="test_hill_effect")
    assert isinstance(hill_effect.half_max_prior, dist.Gamma)
    assert isinstance(hill_effect.slope_prior, dist.HalfNormal)
    assert isinstance(hill_effect.max_effect_prior, dist.Gamma)
    assert hill_effect.effect_mode == "multiplicative"


def test_compute_effect_multiplicative(hill_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.5, 1.0, 1.5])

    with seed(numpyro.handlers.seed, 0):
        result = hill_effect_multiplicative.compute_effect(trend, data)

    half_max, slope, max_effect = 0.5, 1.0, 1.5
    x = _exponent_safe(data / half_max, -slope)  # Calculation of x in Hill function
    expected_effect = max_effect / (1 + x)  # Expected effect based on Hill function
    expected_result = trend * expected_effect  # Multiplicative effect on trend

    assert jnp.allclose(result, expected_result)


def test_compute_effect_additive(hill_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.5, 1.0, 1.5])

    with seed(numpyro.handlers.seed, 0):
        result = hill_effect_additive.compute_effect(trend, data)

    half_max, slope, max_effect = 0.5, 1.0, 1.5
    x = _exponent_safe(data / half_max, -slope)  # Calculation of x in Hill function
    expected_result = max_effect / (1 + x)  # Expected additive effect

    assert jnp.allclose(result, expected_result)
