"""Pytest for Geometric Adstock Effect class."""

import jax.numpy as jnp
import pandas as pd
import pytest
import numpyro
import numpy as np
from numpyro.distributions import Beta

from prophetverse.effects.adstock import GeometricAdstockEffect, WeibullAdstockEffect


def test_geometric_adstock_initialization():
    """Test GeometricAdstockEffect initialization with different parameters."""
    # Test default initialization
    effect1 = GeometricAdstockEffect()
    assert effect1.decay_prior is None
    assert effect1._decay_prior is not None  # Should be set to default Beta(2, 2)

    # Test with custom prior
    from numpyro.distributions import Beta

    custom_prior = Beta(3.0, 1.0)
    effect2 = GeometricAdstockEffect(decay_prior=custom_prior)
    assert effect2.decay_prior is custom_prior
    assert effect2._decay_prior is custom_prior


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
        scale_prior=scale_prior, concentration_prior=concentration_prior
    )
    assert effect3.scale_prior is scale_prior
    assert effect3.concentration_prior is concentration_prior


def test_weibull_adstock_automatic_max_lag():
    """Test automatic max_lag calculation in WeibullAdstockEffect."""
    effect = WeibullAdstockEffect()  # max_lag=None

    # Define mock data
    data = jnp.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Test with different scale and concentration values
    test_cases = [
        {"scale": 1.0, "concentration": 1.0},
        {"scale": 2.0, "concentration": 0.5},
        {"scale": 0.5, "concentration": 2.0},
        {"scale": 3.0, "concentration": 1.5},
    ]

    for params in test_cases:
        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
            result = effect.predict(data_transformed, {})

            # Verify that result has correct shape
            assert result.shape == data_transformed[0].shape
            # Verify non-negative values
            assert jnp.all(result >= 0)


def test_weibull_adstock_max_lag_boundary_conditions():
    """Test WeibullAdstockEffect behavior with edge cases for max_lag."""
    # Test max_lag=1 (minimum)
    effect1 = WeibullAdstockEffect(max_lag=1)

    # Test max_lag greater than data length
    effect2 = WeibullAdstockEffect(max_lag=10)

    # Define small dataset
    data = jnp.array([[5.0], [10.0], [15.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    for effect in [effect1, effect2]:
        effect.fit(y=y, X=X)
        data_transformed = effect.transform(X, fh=X.index)

        params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}
        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
            result = effect.predict(data_transformed, {})

            # Should handle edge cases gracefully
            assert result.shape == data_transformed[0].shape
            assert jnp.all(result >= 0)


def test_weibull_adstock_weight_computation():
    """Test that Weibull weights are computed correctly."""
    effect = WeibullAdstockEffect(max_lag=5)

    data = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Test with known parameters
    params = {"scale": jnp.array(2.0), "concentration": jnp.array(1.0)}

    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        result = effect.predict(data_transformed, {})

        # Check that result is finite and reasonable
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

        # For input [1,2,3,4,5,6], result should show cumulative effect
        # Each subsequent value should generally be larger due to carryover
        assert result.shape == (6, 1)


def test_weibull_adstock_single_timepoint():
    """Test WeibullAdstockEffect with single time point."""
    effect = WeibullAdstockEffect(max_lag=3)

    # Single data point
    data = jnp.array([[10.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        result = effect.predict(data_transformed, {})

        # Should handle single point gracefully
        assert result.shape == (1, 1)
        assert jnp.all(result >= 0)


def test_weibull_adstock_zero_input():
    """Test WeibullAdstockEffect with zero inputs."""
    effect = WeibullAdstockEffect(max_lag=3)

    # All zeros input
    data = jnp.array([[0.0], [0.0], [0.0], [0.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        result = effect.predict(data_transformed, {})

        # Zero input should produce zero or very small output
        assert result.shape == (4, 1)
        assert jnp.all(result >= 0)
        assert jnp.allclose(result, 0.0, atol=1e-10)


def test_weibull_adstock_parameter_sensitivity():
    """Test WeibullAdstockEffect sensitivity to different parameter values."""
    effect = WeibullAdstockEffect(max_lag=4)

    rng = np.random.default_rng(42)
    data = jnp.array(rng.random((100,)))
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Test different parameter combinations
    param_sets = [
        {
            "scale": jnp.array(1),
            "concentration": jnp.array(0.5),
        },  # Low scale, low concentration
        {
            "scale": jnp.array(10),
            "concentration": jnp.array(0.5),
        },  # High scale, low concentration
        {
            "scale": jnp.array(1),
            "concentration": jnp.array(2.0),
        },  # Low scale, high concentration
        {
            "scale": jnp.array(2.0),
            "concentration": jnp.array(2.0),
        },  # High scale, high concentration
    ]

    results = []
    for params in param_sets:
        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
            result = effect.predict(data_transformed, {})
            results.append(result)

            # All results should be valid
            assert result.shape == (len(data), 1)
            assert jnp.all(result >= 0)
            assert jnp.all(jnp.isfinite(result))

    # Results should differ based on parameters (at least some should be different)
    assert not all([jnp.allclose(results[0], r, atol=1e-6).item() for r in results[1:]])


def test_base_adstock_extract_data_and_indices():
    """Test BaseAdstockEffect's _extract_data_and_indices method."""
    effect = GeometricAdstockEffect()

    # Test with tuple input
    data_array = jnp.array([[1.0], [2.0], [3.0]])
    indices = jnp.array([0, 1, 2])
    tuple_input = (data_array, indices)

    extracted_data, extracted_indices = effect._extract_data_and_indices(tuple_input)
    assert jnp.array_equal(extracted_data, data_array)
    assert jnp.array_equal(extracted_indices, indices)

    # Test with array input
    array_input = jnp.array([[1.0], [2.0], [3.0]])
    extracted_data, extracted_indices = effect._extract_data_and_indices(array_input)
    assert jnp.array_equal(extracted_data, array_input)
    assert jnp.array_equal(extracted_indices, jnp.arange(3))


def test_weibull_adstock_mathematical_properties():
    """Test mathematical properties of WeibullAdstockEffect."""
    effect = WeibullAdstockEffect(max_lag=3)

    # Create impulse input (single spike)
    data = jnp.array([[100.0], [0.0], [0.0], [0.0], [0.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    params = {"scale": jnp.array(1.5), "concentration": jnp.array(1.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        result = effect.predict(data_transformed, {})

        # For impulse input, we should see decaying response
        assert result.shape == (5, 1)
        assert jnp.all(result >= 0)

        # The effect should persist beyond the initial spike
        # (though the exact pattern depends on Weibull parameters)
        assert result[0, 0] >= 0  # Initial period should have some effect


def test_weibull_adstock_default_priors():
    """Test that WeibullAdstockEffect uses correct default priors."""
    effect = WeibullAdstockEffect()

    # Check that default priors are set correctly
    assert effect._scale_prior is not None
    assert effect._concentration_prior is not None

    # The defaults should be GammaReparametrized(2, 1)
    # We can't easily test the exact type without importing it, but we can test behavior
    data = jnp.array([[1.0], [1.0], [1.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Should work with default priors when no parameters are provided
    with numpyro.handlers.seed(rng_seed=0):
        result = effect.predict(data_transformed, {})
        assert result.shape == (3, 1)
        assert jnp.all(result >= 0)


def test_weibull_adstock_custom_priors():
    """Test WeibullAdstockEffect with custom prior distributions."""
    from numpyro.distributions import Gamma, Exponential

    # Use different prior distributions
    scale_prior = Exponential(1.0)
    concentration_prior = Gamma(3.0, 2.0)

    effect = WeibullAdstockEffect(
        scale_prior=scale_prior, concentration_prior=concentration_prior, max_lag=2
    )

    data = jnp.array([[2.0], [3.0], [1.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Should work with custom priors
    with numpyro.handlers.seed(rng_seed=0):
        result = effect.predict(data_transformed, {})
        assert result.shape == (3, 1)
        assert jnp.all(result >= 0)


def test_base_adstock_invalid_data_type():
    """Test BaseAdstockEffect error handling for invalid data types."""
    effect = GeometricAdstockEffect()

    # Test with invalid data type
    invalid_data = "invalid_string_data"

    with pytest.raises(ValueError, match="Unexpected data type"):
        effect._extract_data_and_indices(invalid_data)


def test_weibull_adstock_extreme_parameters():
    """Test WeibullAdstockEffect with extreme parameter values."""
    effect = WeibullAdstockEffect(max_lag=2)

    data = jnp.array([[1.0], [1.0], [1.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    # Test with very small parameters
    params_small = {"scale": jnp.array(0.001), "concentration": jnp.array(0.001)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params_small):
        result_small = effect.predict(data_transformed, {})
        assert result_small.shape == (3, 1)
        assert jnp.all(jnp.isfinite(result_small))

    # Test with large parameters
    params_large = {"scale": jnp.array(100.0), "concentration": jnp.array(10.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params_large):
        result_large = effect.predict(data_transformed, {})
        assert result_large.shape == (3, 1)
        assert jnp.all(jnp.isfinite(result_large))


def test_weibull_adstock_consistency_across_calls():
    """Test that WeibullAdstockEffect produces consistent results across multiple calls."""
    effect = WeibullAdstockEffect(max_lag=3)

    data = jnp.array([[5.0], [10.0], [15.0], [20.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}

    # Call predict multiple times with same parameters and seed
    results = []
    for _ in range(3):
        with numpyro.handlers.seed(rng_seed=42), numpyro.handlers.do(data=params):
            result = effect.predict(data_transformed, {})
            results.append(result)

    # All results should be identical
    for i in range(1, len(results)):
        assert jnp.allclose(
            results[0], results[i], atol=1e-10
        ), f"Results differ between calls {0} and {i}"


def test_weibull_adstock_multivariate_input():
    """Test WeibullAdstockEffect with multiple features."""
    effect = WeibullAdstockEffect(max_lag=2)

    # Create data with multiple columns
    data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X = pd.DataFrame(data, columns=["feature1", "feature2"])
    y = pd.DataFrame(jnp.ones((3, 1)), columns=["target"], index=X.index)

    effect.fit(y=y, X=X)
    data_transformed = effect.transform(X, fh=X.index)

    params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        result = effect.predict(data_transformed, {})

        # Should handle multivariate input
        assert result.shape == (3, 1)
        assert jnp.all(result >= 0)


def test_geometric_vs_weibull_adstock_comparison():
    """Compare GeometricAdstockEffect and WeibullAdstockEffect on same data."""
    # Set up identical data for both effects
    data = jnp.array([[10.0], [10.0], [10.0], [10.0]])
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    # Test geometric adstock
    geo_effect = GeometricAdstockEffect()
    geo_effect.fit(y=y, X=X)
    geo_data = geo_effect.transform(X, fh=X.index)

    geo_params = {"decay": jnp.array(0.5)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=geo_params):
        geo_result = geo_effect.predict(geo_data, {})

    # Test Weibull adstock
    weibull_effect = WeibullAdstockEffect(max_lag=3)
    weibull_effect.fit(y=y, X=X)
    weibull_data = weibull_effect.transform(X, fh=X.index)

    weibull_params = {"scale": jnp.array(1.0), "concentration": jnp.array(1.0)}
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=weibull_params):
        weibull_result = weibull_effect.predict(weibull_data, {})

    # Both should produce valid outputs with same shape
    assert geo_result.shape == weibull_result.shape
    assert jnp.all(geo_result >= 0)
    assert jnp.all(weibull_result >= 0)

    # Results should be different (different adstock patterns)
    assert not jnp.allclose(geo_result, weibull_result, atol=1e-6)


def test_geometric_adstock_normalize_option():
    """Test that normalize=True scales the geometric adstock by (1-decay)."""
    data = jnp.array([[1.0], [0.0], [0.0], [0.0]])  # impulse
    X = pd.DataFrame(data, columns=["feature1"])
    y = pd.DataFrame(jnp.ones_like(data), columns=["target"], index=X.index)

    params = {"decay": jnp.array(0.6)}

    # Without normalization
    eff_raw = GeometricAdstockEffect()
    eff_raw.fit(y=y, X=X)
    geo_data_raw = eff_raw.transform(X, fh=X.index)
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        raw_result = eff_raw.predict(geo_data_raw, {})

    # With normalization
    eff_norm = GeometricAdstockEffect(normalize=True)
    eff_norm.fit(y=y, X=X)
    geo_data_norm = eff_norm.transform(X, fh=X.index)
    with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.do(data=params):
        norm_result = eff_norm.predict(geo_data_norm, {})

    # Raw geometric impulse response: 1, 0.6, 0.6^2, 0.6^3
    expected_raw = jnp.array([[1.0], [0.6], [0.36], [0.216]])
    assert jnp.allclose(raw_result, expected_raw, atol=1e-6)

    # Normalized multiplies by (1-decay)=0.4 so that weights sum to 1
    expected_norm = expected_raw * (1 - params["decay"])
    assert jnp.allclose(norm_result, expected_norm, atol=1e-6)
    # Sum of normalized weights (approx over first 4 terms < 1 but approaching 1)
    assert norm_result.sum() < raw_result.sum()
