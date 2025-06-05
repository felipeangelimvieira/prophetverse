import jax
import jax.numpy as jnp
import numpy as np # Import numpy
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS
import pytest 

from src.prophetverse.engine.mcmc import MCMCInferenceEngine, assert_mcmc_converged
from prophetverse.exc import ConvergenceError

# A simple NumPyro model for testing purposes
@pytest.fixture
def dummy_numpyro_model(data=None, future_data=None):
    loc = numpyro.sample("loc", dist.Normal(0, 1))
    scale = numpyro.sample("scale", dist.HalfNormal(1))
    # Check if we are in prediction mode (future_data is not None)
    if future_data is not None:
        # If future_data is provided, we are making predictions
        # For simplicity, let's assume future_data is just a shape for predictions
        with numpyro.plate("pred_plate", future_data.shape[0]):
             return numpyro.sample("pred_obs", dist.Normal(loc, scale))

    # If not in prediction, we are in inference mode (data is not None)
    if data is not None:
        with numpyro.plate("data_plate", data.shape[0]):
            numpyro.sample("obs", dist.Normal(loc, scale), obs=data)


def test_initialization():
    engine = MCMCInferenceEngine()
    assert engine.num_samples == 1000
    assert engine.num_warmup == 200
    assert engine.num_chains == 1
    assert engine.dense_mass == False
    assert engine.r_hat is None
    assert engine.progress_bar == True
    assert engine.summary_ is None
    assert hasattr(engine._rng_key, "shape")  # Basic check for JAX key
    assert jnp.array_equal(engine._rng_key, jax.random.PRNGKey(0)) # Default key is 0


    custom_key = jax.random.PRNGKey(123)
    engine_custom = MCMCInferenceEngine(
        num_samples=500, num_warmup=100, num_chains=2,
        dense_mass=True, rng_key=custom_key, r_hat=1.05, progress_bar=False
    )
    assert engine_custom.num_samples == 500
    assert engine_custom.num_warmup == 100
    assert engine_custom.num_chains == 2
    assert engine_custom.dense_mass == True
    assert jnp.array_equal(engine_custom._rng_key, custom_key)
    assert engine_custom.r_hat == 1.05
    assert engine_custom.progress_bar == False

def test_tags():
    engine = MCMCInferenceEngine()
    assert engine._tags["inference_method"] == "mcmc"

def test_build_kernel():
    engine = MCMCInferenceEngine(dense_mass=True)
    kernel = engine.build_kernel(dummy_numpyro_model)
    assert isinstance(kernel, NUTS)
    assert kernel.model == dummy_numpyro_model
    assert kernel._dense_mass == True  # NUTS stores dense_mass as _dense_mass
    # assert kernel.init_strategy == numpyro.infer.initialization.init_to_mean  # Check specific init strategy

    engine_sparse = MCMCInferenceEngine(dense_mass=False)
    kernel_sparse = engine_sparse.build_kernel(dummy_numpyro_model)
    assert kernel_sparse._dense_mass == False
    
    # Test dense_mass with list of tuples
    dense_mass_list = [('loc',)]
    engine_dense_list = MCMCInferenceEngine(dense_mass=dense_mass_list)
    kernel_dense_list = engine_dense_list.build_kernel(dummy_numpyro_model)
    assert kernel_dense_list._dense_mass == dense_mass_list
