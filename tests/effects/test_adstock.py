"""Pytest for Geometric Adstock Effect class."""

import jax.numpy as jnp
import pandas as pd
import pytest
import numpyro
from numpyro.distributions import Beta

from prophetverse.effects.adstock import GeometricAdstockEffect


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
