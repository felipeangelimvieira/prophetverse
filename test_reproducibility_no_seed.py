"""Test script to reproduce the reproducibility issue WITHOUT explicit seeds."""
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
    # Create simple time series
    dates = pd.period_range(start='2020-01', periods=50, freq='M')
    np.random.seed(42)
    
    # Create trend + noise
    y_values = np.linspace(100, 200, 50) + np.random.normal(0, 5, 50)
    y = pd.DataFrame({'y': y_values}, index=dates)
    
    # Create simple exogenous variable
    X = pd.DataFrame({'x1': np.sin(np.linspace(0, 4*np.pi, 50)) * 10 + 50}, index=dates)
    
    return y, X


def run_model_simulation_no_seed(description=""):
    """Run the model simulation WITHOUT explicit seed and return predictions."""
    print(f"\n{'='*60}")
    print(f"Running: {description} (NO EXPLICIT SEED)")
    print(f"{'='*60}")
    
    # Get dataset
    y, X = create_simple_dataset()
    
    # Create model WITHOUT explicit seed - using defaults
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
            num_steps=50,  # Reduced for faster testing
            # NOTE: NO rng_key parameter passed - should default to jax.random.PRNGKey(0)
        ),
        # NOTE: NO rng_key parameter passed - should default to None (which becomes jax.random.PRNGKey(0))
    )
    
    # Fit and predict
    print("Fitting model...")
    model.fit(y=y, X=X)
    
    print("Making predictions...")
    fh = y.index
    y_pred = model.predict(X=X, fh=fh)
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"First 5 predictions:\n{y_pred.head()}")
    print(f"Mean prediction: {y_pred.mean().values[0]:.6f}")
    print(f"Std prediction: {y_pred.std().values[0]:.6f}")
    
    return y_pred


def test_reproducibility_no_explicit_seed():
    """Test if predictions are the same without explicit seeds."""
    print("\n" + "="*80)
    print("TEST: Reproducibility WITHOUT explicit seeds (using defaults)")
    print("="*80)
    
    pred1 = run_model_simulation_no_seed(description="First run - no explicit seed")
    pred2 = run_model_simulation_no_seed(description="Second run - no explicit seed")
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between runs: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible WITHOUT explicit seeds")
        return True
    else:
        print("✗ TEST FAILED: Predictions differ WITHOUT explicit seeds!")
        print(f"Max difference: {max_diff}")
        print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
        print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")
        print(f"\nDifferences at each time point:")
        print(diff.head(10))
        return False


def test_reproducibility_with_function_wrap_no_seed():
    """Test if predictions are the same when wrapped in a function WITHOUT explicit seeds."""
    print("\n" + "="*80)
    print("TEST: Reproducibility in function WITHOUT explicit seeds (simulating different environments)")
    print("="*80)
    
    def execute_in_isolation():
        """Execute model in isolation to simulate a fresh environment."""
        return run_model_simulation_no_seed(description="Isolated execution - no explicit seed")
    
    pred1 = execute_in_isolation()
    pred2 = execute_in_isolation()
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between isolated runs: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible in isolated execution WITHOUT explicit seeds")
        return True
    else:
        print("✗ TEST FAILED: Predictions differ in isolated execution WITHOUT explicit seeds!")
        print(f"Max difference: {max_diff}")
        print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
        print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")
        print(f"\nDifferences at each time point:")
        print(diff.head(10))
        return False


if __name__ == "__main__":
    # Run both tests WITHOUT explicit seeds
    test1_passed = test_reproducibility_no_explicit_seed()
    test2_passed = test_reproducibility_with_function_wrap_no_seed()
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Test 1 (No explicit seed): {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (Function wrap, no explicit seed): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    print("="*80)
