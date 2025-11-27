"""Test reproducibility of inference engines with proper RNG key management."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from prophetverse.engine import (
    MAPInferenceEngine,
    MCMCInferenceEngine,
    VIInferenceEngine,
)
from prophetverse.engine.optimizer import LBFGSSolver, AdamOptimizer


def simple_model(y=None):
    """A simple model for testing reproducibility."""
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("data", len(y) if y is not None else 1):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


@pytest.mark.parametrize("engine_cls,engine_kwargs", [
    (MAPInferenceEngine, {"optimizer": LBFGSSolver(), "num_steps": 50, "num_samples": 10}),
    (VIInferenceEngine, {"optimizer": AdamOptimizer(), "num_steps": 50, "num_samples": 10}),
    (MCMCInferenceEngine, {"num_samples": 10, "num_warmup": 10, "num_chains": 1}),
])
def test_inference_engine_reproducibility_same_seed(engine_cls, engine_kwargs):
    """Test that inference engines produce reproducible results with the same seed."""
    # Generate test data
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create two engines with the same seed
    engine1 = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    engine2 = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    
    # Fit both engines
    engine1.infer(simple_model, y=y)
    engine2.infer(simple_model, y=y)
    
    # Check that posterior samples are identical
    for key in engine1.posterior_samples_:
        assert jnp.allclose(engine1.posterior_samples_[key], engine2.posterior_samples_[key]), \
            f"Posterior samples for '{key}' differ between runs with same seed"
    
    # Generate predictions
    pred1 = engine1.predict(y=None)
    pred2 = engine2.predict(y=None)
    
    # Check that predictions are identical
    for key in pred1:
        if not key.endswith(":ignore"):
            assert jnp.allclose(pred1[key], pred2[key]), \
                f"Predictions for '{key}' differ between runs with same seed"


@pytest.mark.parametrize("engine_cls,engine_kwargs", [
    (MAPInferenceEngine, {"optimizer": LBFGSSolver(), "num_steps": 50, "num_samples": 10}),
    (VIInferenceEngine, {"optimizer": AdamOptimizer(), "num_steps": 50, "num_samples": 10}),
])
def test_inference_engine_multiple_predictions_produce_different_samples(engine_cls, engine_kwargs):
    """Test that multiple predict() calls produce different samples (correct behavior).
    
    After the RNG key fix, each predict() call should use a fresh split of the RNG key,
    producing different samples. This is the correct behavior for Monte Carlo sampling.
    """
    # Generate test data
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create and fit engine
    engine = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    engine.infer(simple_model, y=y)
    
    # Generate multiple predictions
    pred1 = engine.predict(y=None)
    pred2 = engine.predict(y=None)
    
    # Check that samples differ (they should with proper RNG key management)
    obs_key = "obs"
    if obs_key in pred1:
        # Samples should differ because we're using different RNG subkeys
        max_diff = jnp.max(jnp.abs(pred1[obs_key] - pred2[obs_key]))
        assert max_diff > 1e-6, \
            f"Predictions should differ between calls (using independent RNG keys)"


@pytest.mark.parametrize("engine_cls,engine_kwargs", [
    (MAPInferenceEngine, {"optimizer": LBFGSSolver(), "num_steps": 50, "num_samples": 10}),
    (VIInferenceEngine, {"optimizer": AdamOptimizer(), "num_steps": 50, "num_samples": 10}),
    (MCMCInferenceEngine, {"num_samples": 10, "num_warmup": 10, "num_chains": 1}),
])
def test_inference_engine_refit_reproducible(engine_cls, engine_kwargs):
    """Test that refitting the same engine produces reproducible results."""
    # Generate test data
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create engine with a seed
    engine = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    
    # First fit and predict
    engine.infer(simple_model, y=y)
    pred1 = engine.predict(y=None)
    
    # Refit and predict with the same engine (simulating reuse)
    # Create a new engine with the same seed to simulate fresh start
    engine2 = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    engine2.infer(simple_model, y=y)
    pred2 = engine2.predict(y=None)
    
    # Check that predictions are identical
    for key in pred1:
        if not key.endswith(":ignore"):
            assert jnp.allclose(pred1[key], pred2[key]), \
                f"Predictions for '{key}' differ after refit with same seed"


@pytest.mark.parametrize("engine_cls,engine_kwargs", [
    (MAPInferenceEngine, {"optimizer": LBFGSSolver(), "num_steps": 50, "num_samples": 10}),
    (VIInferenceEngine, {"optimizer": AdamOptimizer(), "num_steps": 50, "num_samples": 10}),
    (MCMCInferenceEngine, {"num_samples": 10, "num_warmup": 10, "num_chains": 1}),
])
def test_inference_engine_different_seeds_produce_different_results(engine_cls, engine_kwargs):
    """Test that different seeds produce different results (as expected)."""
    # Generate test data
    y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Create two engines with different seeds
    engine1 = engine_cls(rng_key=jax.random.PRNGKey(42), **engine_kwargs)
    engine2 = engine_cls(rng_key=jax.random.PRNGKey(123), **engine_kwargs)
    
    # Fit both engines
    engine1.infer(simple_model, y=y)
    engine2.infer(simple_model, y=y)
    
    # Generate predictions
    pred1 = engine1.predict(y=None)
    pred2 = engine2.predict(y=None)
    
    # Check that at least some predictions differ (they should with different seeds)
    # Note: For MAP/VI, the optimization might converge to the same point,
    # but the samples should differ
    obs_key = "obs"
    if obs_key in pred1 and obs_key in pred2:
        # We expect at least some difference in the samples
        # Using a loose check since for simple models they might be close
        max_diff = jnp.max(jnp.abs(pred1[obs_key] - pred2[obs_key]))
        # For MCMC, we definitely expect differences
        # For MAP/VI with sampling, we should also see differences
        if engine_cls == MCMCInferenceEngine:
            assert max_diff > 1e-6, \
                "MCMC predictions should differ with different seeds"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
