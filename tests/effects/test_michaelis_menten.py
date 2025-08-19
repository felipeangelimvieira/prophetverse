import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.handlers import seed

from prophetverse.effects.michaelis_menten import MichaelisMentenEffect


@pytest.fixture
def michaelis_menten_effect_multiplicative():
    return MichaelisMentenEffect(
        max_effect_prior=dist.Delta(2.0),
        half_saturation_prior=dist.Delta(1.0),
        effect_mode="multiplicative",
    )


@pytest.fixture
def michaelis_menten_effect_additive():
    return MichaelisMentenEffect(
        max_effect_prior=dist.Delta(2.0),
        half_saturation_prior=dist.Delta(1.0),
        effect_mode="additive",
    )


def test_initialization_defaults():
    mm_effect = MichaelisMentenEffect()
    assert isinstance(mm_effect._max_effect_prior, dist.Gamma)
    assert isinstance(mm_effect._half_saturation_prior, dist.Gamma)
    assert mm_effect.effect_mode == "multiplicative"


def test__predict_multiplicative(michaelis_menten_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([0.5, 1.0, 2.0]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = michaelis_menten_effect_multiplicative.predict(
            data=data, predicted_effects={"trend": trend}
        )

    max_effect, half_saturation = 2.0, 1.0
    # Michaelis-Menten equation: effect = (max_effect * data) / (half_saturation + data)
    expected_effect = (max_effect * data) / (half_saturation + data)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__predict_additive(michaelis_menten_effect_additive):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([0.5, 1.0, 2.0]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = michaelis_menten_effect_additive.predict(
            data=data, predicted_effects={"trend": trend}
        )

    max_effect, half_saturation = 2.0, 1.0
    # Michaelis-Menten equation: effect = (max_effect * data) / (half_saturation + data)
    expected_result = (max_effect * data) / (half_saturation + data)

    assert jnp.allclose(result, expected_result)


def test__predict_with_zero_data(michaelis_menten_effect_multiplicative):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([0.0, 0.0, 0.0]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = michaelis_menten_effect_multiplicative.predict(
            data=data, predicted_effects={"trend": trend}
        )

    max_effect, half_saturation = 2.0, 1.0
    # With clipping, data becomes 1e-9, so effect = (2.0 * 1e-9) / (1.0 + 1e-9) ≈ 2e-9
    clipped_data = jnp.clip(data, 1e-9, None)
    expected_effect = (max_effect * clipped_data) / (half_saturation + clipped_data)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test__predict_with_empty_data(michaelis_menten_effect_multiplicative):
    trend = jnp.array([]).reshape((-1, 1))
    data = jnp.array([]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = michaelis_menten_effect_multiplicative.predict(
            data=data, predicted_effects={"trend": trend}
        )

    max_effect, half_saturation = 2.0, 1.0
    expected_effect = (max_effect * data) / (half_saturation + data)
    expected_result = trend * expected_effect

    assert jnp.allclose(result, expected_result)


def test_michaelis_menten_saturation_behavior():
    """Test that the effect approaches max_effect as data approaches infinity."""
    mm_effect = MichaelisMentenEffect(
        max_effect_prior=dist.Delta(10.0),
        half_saturation_prior=dist.Delta(1.0),
        effect_mode="additive",
    )
    
    # Test with very large data values
    trend = jnp.array([1.0]).reshape((-1, 1))
    large_data = jnp.array([1000.0]).reshape((-1, 1))

    with seed(numpyro.handlers.seed, 0):
        result = mm_effect.predict(
            data=large_data, predicted_effects={"trend": trend}
        )

    # For large data, (max_effect * data) / (half_saturation + data) ≈ max_effect
    # Since data >> half_saturation: effect ≈ 10.0 * 1000 / 1001 ≈ 9.99
    expected_result = jnp.array([[9.99]])
    assert jnp.allclose(result, expected_result, atol=1e-2)


def test_michaelis_menten_half_saturation_point():
    """Test that at half_saturation point, effect equals max_effect/2."""
    mm_effect = MichaelisMentenEffect(
        max_effect_prior=dist.Delta(10.0),
        half_saturation_prior=dist.Delta(2.0),
        effect_mode="additive",
    )
    
    trend = jnp.array([1.0]).reshape((-1, 1))
    half_sat_data = jnp.array([2.0]).reshape((-1, 1))  # Equal to half_saturation

    with seed(numpyro.handlers.seed, 0):
        result = mm_effect.predict(
            data=half_sat_data, predicted_effects={"trend": trend}
        )

    # At half_saturation: effect = (10.0 * 2.0) / (2.0 + 2.0) = 20.0 / 4.0 = 5.0
    expected_result = jnp.array([[5.0]])
    assert jnp.allclose(result, expected_result)