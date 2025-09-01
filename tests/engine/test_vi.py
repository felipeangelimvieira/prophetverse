"""Tests for VIInferenceEngine."""

import pytest
import jax.numpy as jnp
import numpy as np
import numpyro

from prophetverse.engine.vi import VIInferenceEngine, VIInferenceEngineError


def _model(obs):
    """Simple test model."""
    mean = numpyro.sample("mean", numpyro.distributions.Normal(0, 1))
    return numpyro.sample("y", numpyro.distributions.Normal(mean, 1), obs=obs)


class TestVIInferenceEngine:
    """Test VIInferenceEngine functionality."""

    def test_guide_parameter_validation(self):
        """Test that invalid guide parameter raises error."""
        with pytest.raises(ValueError, match="Unknown guide"):
            VIInferenceEngine(guide="InvalidGuide")

    def test_available_guides(self):
        """Test that all available guides can be instantiated."""
        available_guides = ["AutoNormal", "AutoMultivariateNormal", 
                          "AutoDiagonalNormal", "AutoLowRankMultivariateNormal"]
        
        for guide in available_guides:
            engine = VIInferenceEngine(guide=guide, num_steps=10)
            assert engine.guide == guide

    def test_inference_with_different_guides(self):
        """Test inference converges with different guides."""
        obs = jnp.array(np.random.normal(0, 1, 100))
        guides_to_test = ["AutoNormal", "AutoDiagonalNormal"]
        
        for guide in guides_to_test:
            engine = VIInferenceEngine(guide=guide, num_steps=100)
            engine.infer(_model, obs=obs)
            
            assert isinstance(engine.posterior_samples_, dict)
            assert "mean" in engine.posterior_samples_
            assert jnp.isfinite(engine.posterior_samples_["mean"].mean().item())

    def test_prediction(self):
        """Test prediction after inference."""
        obs = jnp.array(np.random.normal(0, 1, 50))
        engine = VIInferenceEngine(guide="AutoNormal", num_steps=100)
        engine.infer(_model, obs=obs)
        
        predictions = engine.predict(obs=None)
        assert isinstance(predictions, dict)
        assert "y" in predictions

    def test_nan_loss_error(self):
        """Test that NaN loss raises appropriate error."""
        engine = VIInferenceEngine(guide="AutoNormal")
        
        # Create mock run results with NaN loss
        class MockRunResults:
            losses = jnp.array([1.0, 2.0, jnp.nan])
        
        with pytest.raises(VIInferenceEngineError, match="NaN losses in VIInferenceEngine"):
            engine.raise_error_if_nan_loss(MockRunResults())

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        engine = VIInferenceEngine()
        
        assert engine.guide == "AutoNormal"
        assert engine.num_steps == 10_000
        assert engine.num_samples == 1000
        assert not engine.progress_bar
        assert not engine.stable_update
        assert not engine.forward_mode_differentiation

    def test_tags(self):
        """Test that engine has correct tags."""
        engine = VIInferenceEngine()
        assert engine._tags["inference_method"] == "vi"

    def test_get_test_params(self):
        """Test that get_test_params returns valid configurations."""
        test_params = VIInferenceEngine.get_test_params()
        
        assert isinstance(test_params, list)
        assert len(test_params) >= 1
        
        for params in test_params:
            engine = VIInferenceEngine(**params)
            assert engine.guide in ["AutoNormal", "AutoMultivariateNormal", "AutoDiagonalNormal"]