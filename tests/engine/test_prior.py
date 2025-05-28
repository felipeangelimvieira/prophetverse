import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import pytest

from src.prophetverse.engine.prior import PriorPredictiveInferenceEngine

# A simple NumPyro model for testing purposes
@pytest.fixture
def dummy_prior_model(data=None, future_data=None, some_param_to_sub=1):
    loc = numpyro.sample("loc", dist.Normal(0, 1))
    scale = numpyro.sample("scale", dist.HalfNormal(some_param_to_sub)) # Parameter that can be substituted
    
    # For _infer, we are interested in prior samples, not necessarily "obs"
    # For _predict, we might generate "obs" or "pred_obs"
    
    if future_data is not None: # Prediction mode
        with numpyro.plate("pred_plate", future_data.shape[0]):
            return numpyro.sample("pred_obs", dist.Normal(loc, scale))
    elif data is not None: # Inference mode (though for prior, obs is usually not passed or ignored)
         with numpyro.plate("data_plate", data.shape[0]):
            # Prior engine removes "obs" from posterior_samples_
            numpyro.sample("obs", dist.Normal(loc, scale), obs=data) 
    else: # Pure prior sampling of loc and scale
        pass



def test_initialization():
    engine = PriorPredictiveInferenceEngine()
    assert engine.num_samples == 1000
    assert engine.substitute is None
    assert hasattr(engine._rng_key, "shape")  # Basic check for JAX key

    custom_key = jax.random.PRNGKey(123)
    sub_data = {"some_param_to_sub": 2.0}
    engine_custom = PriorPredictiveInferenceEngine(
        num_samples=50, rng_key=custom_key, substitute=sub_data
    )
    assert engine_custom.num_samples == 50
    assert jnp.array_equal(engine_custom._rng_key, custom_key)
    assert engine_custom.substitute == sub_data

def test_tags():
    engine = PriorPredictiveInferenceEngine()
    assert engine._tags["inference_method"] == "prior_predictive"
