"""Test script to check if calling fit multiple times causes reproducibility issues."""
import sys
sys.path.insert(0, '/home/runner/work/prophetverse/prophetverse/src')

import numpy as np
import pandas as pd
import jax
from prophetverse.sktime import Prophetverse
from prophetverse.effects.trend import PiecewiseLinearTrend
from prophetverse.effects import LinearEffect
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import LBFGSSolver


def create_simple_dataset():
    """Create a simple dataset for testing."""
    dates = pd.period_range(start='2020-01', periods=50, freq='M')
    np.random.seed(42)
    
    y_values = np.linspace(100, 200, 50) + np.random.normal(0, 5, 50)
    y = pd.DataFrame({'y': y_values}, index=dates)
    
    X = pd.DataFrame({'x1': np.sin(np.linspace(0, 4*np.pi, 50)) * 10 + 50}, index=dates)
    
    return y, X


def test_refit_same_model():
    """Test if calling fit() multiple times on the SAME model instance produces the same results."""
    print("\n" + "="*80)
    print("TEST: Reproducibility when calling fit() multiple times on same model instance")
    print("="*80)
    
    y, X = create_simple_dataset()
    
    # Create model with explicit seed
    model = Prophetverse(
        trend=PiecewiseLinearTrend(
            changepoint_interval=10,
            changepoint_prior_scale=0.1,
        ),
        exogenous_effects=[
            ("x1_effect", LinearEffect(), "x1")
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=LBFGSSolver(
                memory_size=100,
                max_linesearch_steps=100,
                learning_rate=0.001
            ),
            num_steps=50,
            rng_key=jax.random.PRNGKey(42),
        ),
        rng_key=jax.random.PRNGKey(42),
    )
    
    print("\nFirst fit()...")
    model.fit(y=y, X=X)
    pred1 = model.predict(X=X, fh=y.index)
    print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
    
    print("\nSecond fit() - on the SAME model instance...")
    model.fit(y=y, X=X)
    pred2 = model.predict(X=X, fh=y.index)
    print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between fits: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible when refitting same model")
        return True
    else:
        print("✗ TEST FAILED: Predictions differ when refitting same model!")
        print(f"Max difference: {max_diff}")
        print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
        print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")
        return False


def test_new_model_instances():
    """Test if creating new model instances with same seed produces the same results."""
    print("\n" + "="*80)
    print("TEST: Reproducibility with NEW model instances (same seed)")
    print("="*80)
    
    y, X = create_simple_dataset()
    
    # Create first model
    print("\nCreating first model instance...")
    model1 = Prophetverse(
        trend=PiecewiseLinearTrend(
            changepoint_interval=10,
            changepoint_prior_scale=0.1,
        ),
        exogenous_effects=[
            ("x1_effect", LinearEffect(), "x1")
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=LBFGSSolver(
                memory_size=100,
                max_linesearch_steps=100,
                learning_rate=0.001
            ),
            num_steps=50,
            rng_key=jax.random.PRNGKey(42),
        ),
        rng_key=jax.random.PRNGKey(42),
    )
    model1.fit(y=y, X=X)
    pred1 = model1.predict(X=X, fh=y.index)
    print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
    
    # Create second model (new instance, same configuration)
    print("\nCreating second model instance (same configuration)...")
    model2 = Prophetverse(
        trend=PiecewiseLinearTrend(
            changepoint_interval=10,
            changepoint_prior_scale=0.1,
        ),
        exogenous_effects=[
            ("x1_effect", LinearEffect(), "x1")
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=LBFGSSolver(
                memory_size=100,
                max_linesearch_steps=100,
                learning_rate=0.001
            ),
            num_steps=50,
            rng_key=jax.random.PRNGKey(42),
        ),
        rng_key=jax.random.PRNGKey(42),
    )
    model2.fit(y=y, X=X)
    pred2 = model2.predict(X=X, fh=y.index)
    print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between new instances: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible with new model instances")
        return True
    else:
        print("✗ TEST FAILED: Predictions differ with new model instances!")
        print(f"Max difference: {max_diff}")
        return False


if __name__ == "__main__":
    test1_passed = test_refit_same_model()
    test2_passed = test_new_model_instances()
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Test 1 (Refit same model): {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (New model instances): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    print("="*80)
