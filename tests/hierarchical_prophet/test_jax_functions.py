import jax.numpy as jnp
import pytest

from hierarchical_prophet.jax_functions import (additive_mean_model,
                                           get_changepoint_offset_adjustment,
                                           get_changepoint_slopes,
                                           multiplicative_mean_model)


def test_additive_mean_model():
    trend = jnp.array([1, 2, 3])
    args = (jnp.array([4, 5, 6]), jnp.array([7, 8, 9]))
    result = additive_mean_model(trend, *args)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == trend.shape


def test_multiplicative_mean_model():
    trend = jnp.array([1, 2, 3])
    args = (jnp.array([4, 5, 6]), jnp.array([7, 8, 9]))
    result = multiplicative_mean_model(trend, *args)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == trend.shape
