"""Pytest for Geometric Adstock Effect class."""

import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro import handlers
from numpyro.distributions import Beta

from prophetverse.effects.adstock import GeometricAdstockEffect


def test_geometric_adstock_sampling():
    """Test parameter sampling using numpyro.handlers.trace."""
    effect = GeometricAdstockEffect(decay_prior=Beta(2, 2))
    data = jnp.ones((10, 1))  # Dummy data
    predicted_effects = {}

    with handlers.trace() as trace, handlers.seed(rng_seed=0):
        effect._sample_params(data, predicted_effects)

    # Verify trace contains decay site
    assert "decay" in trace, "Decay parameter not found in trace."

    # Verify decay is sampled from the correct prior
    assert trace["decay"]["type"] == "sample", "Decay parameter not sampled."
    assert isinstance(
        trace["decay"]["fn"], Beta
    ), "Decay parameter not sampled from Beta distribution."


def test_geometric_adstock_predict():
    """Test the predict method for correctness with predefined parameters."""
    effect = GeometricAdstockEffect()

    # Define mock data and parameters
    data = jnp.array([[10.0], [20.0], [30.0]])  # Example input data (T, 1)
    params = {"decay": jnp.array(0.5)}
    predicted_effects = {}

    # Call _predict
    result = effect._predict(data, predicted_effects, params)

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
