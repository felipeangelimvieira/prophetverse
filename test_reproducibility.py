"""Test script to reproduce the reproducibility issue."""
import sys
sys.path.insert(0, '/home/runner/work/prophetverse/prophetverse/src')

import numpy as np
import pandas as pd
import jax
from prophetverse.sktime import Prophetverse
from prophetverse.effects.trend import PiecewiseLogisticTrend
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import LBFGSSolver
from prophetverse.datasets._mmm.dataset1 import get_dataset


def run_model_simulation(description=""):
    """Run the model simulation and return predictions."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    # Get dataset
    yy, Xx, lift_test, true_effect, model2 = get_dataset()
    
    # Create model with explicit seed
    model337 = Prophetverse(
        trend=PiecewiseLogisticTrend(
            changepoint_prior_scale=0.1,
            changepoint_interval=8,
            changepoint_range=-8,
        ),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(
                    sp_list=["YE"],
                    fourier_terms_list=[1],
                    freq="Q",
                    prior_scale=0.1,
                    effect_mode="multiplicative",
                ),
                None,
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=LBFGSSolver(
                memory_size=1000,
                max_linesearch_steps=1000,
                learning_rate=0.001
            ),
            num_steps=100,  # Reduced for faster testing
            rng_key=jax.random.PRNGKey(42),  # Explicit seed
        ),
        rng_key=jax.random.PRNGKey(42),  # Explicit seed
    )
    
    # Fit and predict
    model337.fit(y=yy, X=Xx)
    
    fh = yy.index.get_level_values(-1).unique()
    y_pred = model337.predict(X=Xx, fh=fh)
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"First 5 predictions:\n{y_pred.head()}")
    print(f"Mean prediction: {y_pred.mean().values[0]:.6f}")
    print(f"Std prediction: {y_pred.std().values[0]:.6f}")
    
    return y_pred


def test_reproducibility_same_execution():
    """Test if predictions are the same within the same execution."""
    print("\n" + "="*80)
    print("TEST 1: Reproducibility in same execution")
    print("="*80)
    
    pred1 = run_model_simulation("First run - same execution")
    pred2 = run_model_simulation("Second run - same execution")
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between runs: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible in same execution")
    else:
        print("✗ TEST FAILED: Predictions differ in same execution!")
        print(f"Max difference: {max_diff}")
        print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
        print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")


def test_reproducibility_with_function_wrap():
    """Test if predictions are the same when wrapped in a function (simulating different environments)."""
    print("\n" + "="*80)
    print("TEST 2: Reproducibility when wrapped in function (simulating different environments)")
    print("="*80)
    
    def execute_in_isolation():
        """Execute model in isolation to simulate a fresh environment."""
        return run_model_simulation("Isolated execution")
    
    pred1 = execute_in_isolation()
    pred2 = execute_in_isolation()
    
    # Check if predictions are identical
    diff = (pred1 - pred2).abs()
    max_diff = diff.max().values[0]
    
    print(f"\n{'='*60}")
    print(f"Maximum difference between isolated runs: {max_diff}")
    print(f"{'='*60}")
    
    if max_diff < 1e-6:
        print("✓ TEST PASSED: Predictions are reproducible in isolated execution")
    else:
        print("✗ TEST FAILED: Predictions differ in isolated execution!")
        print(f"Max difference: {max_diff}")
        print(f"Predictions 1 mean: {pred1.mean().values[0]:.6f}")
        print(f"Predictions 2 mean: {pred2.mean().values[0]:.6f}")


if __name__ == "__main__":
    # Run both tests
    test_reproducibility_same_execution()
    test_reproducibility_with_function_wrap()
    
    print("\n" + "="*80)
    print("All tests completed")
    print("="*80)
