from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro import distributions as dist

from prophetverse.effects import (AbstractEffect, LinearEffect, LogEffect,
                                  additive_effect, multiplicative_effect)


# Mock for numpyro.sample
@pytest.fixture
def mock_numpyro_sample():
    with patch("numpyro.sample") as mock:
        yield mock


# Fixtures for creating effect instances
@pytest.fixture
def log_effect():
    return LogEffect(id="log_test")


@pytest.fixture
def linear_effect():
    return LinearEffect(id="linear_test")




# Testing AbstractEffect instantiation
def test_abstract_effect_cannot_be_instantiated():
    with pytest.raises(TypeError):
        effect = AbstractEffect()


# Testing LogEffect
def test_log_effect_compute(mock_numpyro_sample, log_effect):
    mock_numpyro_sample.return_value = (
        1  # Fix the sample return value for predictability
    )
    trend = jnp.array([1.0])
    data = jnp.array([2.0])
    effect = log_effect.compute_effect(trend, data)
    assert effect.shape == trend.shape
    mock_numpyro_sample.assert_called()


# Testing LinearEffect
def test_linear_effect_compute(mock_numpyro_sample, linear_effect):
    mock_numpyro_sample.return_value = jnp.array([1.0])
    trend = jnp.array([[10.0]])
    data = jnp.array([[2.0]])
    effect = linear_effect.compute_effect(trend, data)
    assert effect.shape == trend.shape
    mock_numpyro_sample.assert_called()




# Testing Additive and Multiplicative effects
def test_additive_effect():
    trend = jnp.array([10.0])
    data = jnp.array([2.0])
    coefficients = jnp.array([1.0])
    result = additive_effect(trend, data, coefficients)
    assert result == pytest.approx(2.0)


def test_multiplicative_effect():
    trend = jnp.array([10.0])
    data = jnp.array([2.0])
    coefficients = jnp.array([1.0])
    result = multiplicative_effect(trend, data, coefficients)
    assert result == pytest.approx(20.0)
