"""Pytest for Geometric Adstock Effect class."""

import jax.numpy as jnp
import pandas as pd
import pytest
import numpyro
from numpyro.distributions import Beta

from prophetverse.effects.adstock import GeometricAdstockEffect, WeibullAdstockEffect


def test_geometric_adstock_predict():
    """Test the predict method for correctness with predefined parameters."""
    effect = GeometricAdstockEffect()

    # Define mock data and parameters
    data = jnp.array([[10.0], [20.0], [30.0]])  # Example input data (T, 1)
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)
    params = {"decay": jnp.array(0.5)}
    predicted_effects = {}

    effect.fit(y=y, X=X)
    data = effect.transform(X, fh=X.index)
    # Call _predict
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        with numpyro.handlers.trace() as exec_trace:
            result = effect.predict(data, predicted_effects)

    # Expected adstock output
    expected = jnp.array(
        [
            [10.0],
            [20.0 + 0.5 * 10.0],
            [30.0 + 0.5 * (20.0 + 0.5 * 10.0)],
        ]
    )

    # Verify output shape
    assert result.shape == data[0].shape, "Output shape mismatch."

    # Verify output values
    assert jnp.allclose(result, expected), "Adstock computation incorrect."

    ## Now test calling adstock with a different time span
    X2 = X.iloc[2:]
    data = effect.transform(X2, fh=X2.index)
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        with numpyro.handlers.trace() as exec_trace:
            result2 = effect.predict(data, predicted_effects)
    # Expected adstock output
    assert jnp.allclose(
        result2, expected[2:]
    ), "Adstock computation incorrect for different time span."


def test_weibull_adstock_predict():
    """Test the predict method for WeibullAdstockEffect."""
    effect = WeibullAdstockEffect(max_lag=3)

    # Define mock data and parameters
    data = jnp.array([[10.0], [20.0], [30.0], [40.0]])  # Example input data (T, 1)
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)
    params = {"scale": jnp.array(2.0), "concentration": jnp.array(1.5)}
    predicted_effects = {}

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)
    
    # Call _predict
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        with numpyro.handlers.trace() as exec_trace:
            result = effect.predict(data_transformed, predicted_effects)

    # Verify output shape
    assert result.shape == data_transformed[0].shape, "Output shape mismatch."
    
    # Verify output is non-negative (adstock should not be negative)
    assert jnp.all(result >= 0), "Adstock values should be non-negative."
    
    # Test different time span like in geometric test
    X2 = X.iloc[2:]
    data_transformed2 = effect.transform(X2, fh=X2.index)
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        with numpyro.handlers.trace() as exec_trace:
            result2 = effect.predict(data_transformed2, predicted_effects)
    
    # Verify shape consistency
    assert result2.shape[0] == 2, "Result should have 2 time steps."
    assert jnp.allclose(
        result2, result[2:]
    ), "Weibull adstock computation incorrect for different time span."


def test_weibull_adstock_initialization():
    """Test WeibullAdstockEffect initialization with different parameters."""
    # Test default initialization
    effect1 = WeibullAdstockEffect()
    assert effect1.max_lag is None
    assert effect1.scale_prior is None
    assert effect1.concentration_prior is None
    
    # Test with custom max_lag
    effect2 = WeibullAdstockEffect(max_lag=5)
    assert effect2.max_lag == 5
    
    # Test with custom priors
    from numpyro.distributions import Gamma
    scale_prior = Gamma(1.0, 1.0)
    concentration_prior = Gamma(2.0, 1.0)
    
    effect3 = WeibullAdstockEffect(
        scale_prior=scale_prior,
        concentration_prior=concentration_prior
    )
    assert effect3.scale_prior is scale_prior
    assert effect3.concentration_prior is concentration_prior
