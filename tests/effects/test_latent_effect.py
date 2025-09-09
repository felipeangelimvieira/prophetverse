"""Test for latent effect functionality."""
import pytest
import pandas as pd
import numpy as np
from prophetverse.sktime import Prophetverse
from prophetverse.effects.linear import LinearEffect
from prophetverse.effects.fourier import LinearFourierSeasonality
from prophetverse.effects.trend import FlatTrend
from prophetverse.engine.map import MAPInferenceEngine
from prophetverse.engine.optimizer import AdamOptimizer
from prophetverse.utils.regex import no_input_columns, starts_with


def make_test_data():
    """Create simple test data."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    y = pd.Series(np.random.randn(30) + 10, index=dates)
    X = pd.DataFrame({
        "feature1": np.random.randn(30),
        "feature2": np.random.randn(30),
    }, index=dates)
    return y, X


@pytest.mark.smoke  
def test_latent_effect_basic():
    """Test that latent effects are computed but excluded from final mean."""
    y, X = make_test_data()
    
    # Create a model with a latent effect and a regular effect
    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            # Regular seasonality effect - should be included in final mean
            (
                "seasonality",
                LinearFourierSeasonality(sp_list=[7], fourier_terms_list=[1], freq="D"),
                no_input_columns,
            ),
            # Latent effect - should be computed but excluded from final mean
            (
                "latent/hidden_seasonality", 
                LinearFourierSeasonality(sp_list=[7], fourier_terms_list=[1], freq="D"),
                no_input_columns,
            ),
            # Regular feature effect - should be included in final mean
            (
                "feature1_effect",
                LinearEffect(),
                starts_with("feature1"),
            ),
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), 
            num_steps=5, 
            num_samples=1
        ),
    )
    
    # Fit the model
    forecaster.fit(y, X)
    
    # Get predictions and components - use relative forecasting horizon
    fh = [1, 2, 3]
    # Create future X data for prediction
    future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=3, freq="D")
    X_future = pd.DataFrame({
        "feature1": np.random.randn(3),
        "feature2": np.random.randn(3),
    }, index=future_dates)
    
    y_pred = forecaster.predict(X=X_future, fh=fh)
    components = forecaster.predict_components(X=X_future, fh=fh)
    
    # Verify basic functionality
    assert y_pred.shape == (3,)  # univariate series
    assert len(components.columns) >= 3  # Should have at least trend, seasonality, feature1_effect
    
    # The key test: latent effect should be in components but not included in the mean
    # We'll verify this by checking that the mean approximately equals non-latent effects
    print("Components columns:", components.columns.tolist())
    print("Components:")
    print(components.to_string())
    
    # Get all non-latent effects
    non_latent_effects = []
    for col in components.columns:
        if not col.startswith("latent/") and col not in ["mean", "obs"]:
            non_latent_effects.append(col)
    
    print(f"Non-latent effects: {non_latent_effects}")
    
    # Sum up all non-latent effects
    expected_mean = sum(components[col].values for col in non_latent_effects)
    
    # Get the actual mean from components (this is the final output)
    actual_mean = y_pred.values
    
    print("Expected mean (sum of non-latent effects):", expected_mean)
    print("Actual y_pred:", actual_mean)
    print("Difference:", actual_mean - expected_mean)
    
    # They should be approximately equal (allowing for numerical precision)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-4, atol=1e-4)
    
    # Verify that latent effect was computed and is available in components
    assert "latent/hidden_seasonality" in components.columns
    latent_effect = components["latent/hidden_seasonality"].values
    
    # The latent effect might be zero due to limited training steps, but it should exist
    # This confirms that latent effects are computed and available to other effects
    print("Latent effect values:", latent_effect)
    
    print("✓ Latent effect test passed: latent effects computed but excluded from final mean")


@pytest.mark.smoke
def test_latent_effect_available_to_other_effects():
    """Test that latent effects are available to other effects via predicted_effects."""
    # This is a more complex test that would require a custom effect
    # For now, we just verify the basic functionality works
    y, X = make_test_data()
    
    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            (
                "latent/base_effect",
                LinearEffect(),
                starts_with("feature1"),
            ),
            (
                "regular_effect", 
                LinearEffect(),
                starts_with("feature2"),
            ),
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), 
            num_steps=5, 
            num_samples=1
        ),
    )
    
    forecaster.fit(y, X)
    # Create future X data for prediction
    future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=2, freq="D")
    X_future = pd.DataFrame({
        "feature1": np.random.randn(2),
        "feature2": np.random.randn(2),
    }, index=future_dates)
    components = forecaster.predict_components(X=X_future, fh=[1, 2])
    
    # Both effects should be computed and available in components
    assert "latent/base_effect" in components.columns
    assert "regular_effect" in components.columns
    
    # The latent effect should not contribute to the final prediction
    # But this test mainly verifies the basic structure works
    print("✓ Latent effects are available in predicted components")


if __name__ == "__main__":
    test_latent_effect_basic()
    test_latent_effect_available_to_other_effects()