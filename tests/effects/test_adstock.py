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
    params = {"decay": jnp.array(0.5)}
    predicted_effects = {}

    # Call _predict
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        with numpyro.handlers.trace() as exec_trace:

            result = effect.predict(data, predicted_effects, params)

    # Expected adstock output
    expected = jnp.array(
        [
            [10.0],
            [20.0 + 0.5 * 10.0],
            [30.0 + 0.5 * (20.0 + 0.5 * 10.0)],
        ]
    )

    # Verify output shape
    assert result.shape == data.shape, "Output shape mismatch."

    # Verify output values
    assert jnp.allclose(result, expected), "Adstock computation incorrect."


def test_error_when_different_fh():
    effect = GeometricAdstockEffect()
    X = pd.DataFrame(
        data={"exog": [10.0, 20.0, 30.0, 30.0, 40.0, 50.0]},
        index=pd.date_range("2021-01-01", periods=6),
    )
    fh = X.index
    effect.transform(X=X, fh=fh)

    effect.transform(X=X.iloc[:1], fh=fh[:1])
    with pytest.raises(ValueError):
        effect.transform(X=X, fh=fh[1:])
