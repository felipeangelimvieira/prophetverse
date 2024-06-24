import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from numpyro import distributions as dist

from prophetverse.effects.base import AbstractEffect
from prophetverse.effects.effects import (
    HillEffect,
    LinearEffect,
    LogEffect,
    additive_effect,
    multiplicative_effect,
)
from prophetverse.utils.regex import exact


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "x1": range(10),
            "x2": range(10, 20),
            "log_x1": [0.1 * i for i in range(10)],
            "lin_x2": [0.2 * i for i in range(10, 20)],
        }
    )


@pytest.fixture
def effects():
    return [
        LogEffect(
            id="log1",
            regex=r"log_.*",
            scale_prior=dist.Gamma(1, 1),
            rate_prior=dist.Gamma(1, 1),
        ),
        LinearEffect(id="lin1", regex=r"lin_.*", prior=(dist.Normal, 0, 1)),
        HillEffect(
            id="hill1",
            half_max_prior=dist.Gamma(1, 1),
            slope_prior=dist.HalfNormal(10),
            max_effect_prior=dist.Gamma(1, 1),
            regex=exact("x1"),
        ),
    ]


# Test initialization and attribute assignment
@pytest.mark.parametrize(
    "effect_class, attributes",
    [
        (
            LogEffect,
            {
                "id": "log1",
                "effect_mode": "multiplicative",
                "scale_prior": dist.Gamma(1, 1),
                "rate_prior": dist.Gamma(1, 1),
            },
        ),
        (
            LinearEffect,
            {"id": "lin1", "effect_mode": "additive", "prior": (dist.Normal, 0, 1)},
        ),
        (
            HillEffect,
            {
                "id": "hill1",
                "effect_mode": "multiplicative",
                "half_max_prior": dist.Gamma(1, 1),
                "slope_prior": dist.HalfNormal(10),
                "max_effect_prior": dist.Gamma(1, 1),
            },
        ),
    ],
)
def test_effect_initialization(effect_class, attributes):
    effect = effect_class(**attributes)
    for attr, value in attributes.items():
        assert getattr(effect, attr) == value


# Test column matching functionality
def test_match_columns(sample_data):
    log_effect = LogEffect(id="log1", regex=r"log_.*")
    matched_columns = log_effect.match_columns(sample_data.columns)
    assert "log_x1" in matched_columns


# Test data splitting
def test_split_data_into_effects(effects, sample_data):
    split_data = AbstractEffect.split_data_into_effects(sample_data, effects)
    assert all(effect.id in split_data for effect in effects)


# Test compute_effect for each effect
@pytest.mark.parametrize(
    "effect, data, trend, expected_shape",
    [
        (
            LogEffect(
                id="test_log",
                regex=r"log_.*",
                scale_prior=dist.Gamma(1, 1),
                rate_prior=dist.Gamma(1, 1),
            ),
            jnp.array([1, 2, 3]).reshape((-1, 1)),
            jnp.array([100, 200, 300]).reshape((-1, 1)),
            (3, 1),
        ),
        (
            LinearEffect(id="test_lin", regex=r"lin_.*", prior=dist.Normal(0, 1)),
            jnp.array([[1, 2], [3, 4], [5, 6]]).reshape((-1, 2)),
            jnp.array([100, 200, 300]).reshape((-1, 1)),
            (3, 1),
        ),
        (
            HillEffect(
                id="test_hill",
                half_max_prior=dist.Gamma(1, 1),
                slope_prior=dist.HalfNormal(10),
                max_effect_prior=dist.Gamma(1, 1),
            ),
            jnp.array([1, 2, 3]).reshape((-1, 1)),
            jnp.array([100, 200, 300]).reshape((-1, 1)),
            (3, 1),
        ),
    ],
)
def test_compute_effect(effect: AbstractEffect, data, trend, expected_shape):

    with numpyro.handlers.seed(rng_seed=0):
        result = effect.compute_effect(trend, data)
    assert result.shape == expected_shape
    assert jnp.all(~jnp.isnan(result))


# Testing AbstractEffect instantiation
def test_abstract_effect_cannot_be_instantiated():
    with pytest.raises(TypeError):
        effect = AbstractEffect()


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
