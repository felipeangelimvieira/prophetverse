import jax.numpy as jnp
import jax
import pytest
from numpyro.infer.svi import SVIRunResult

from prophetverse.engine import MAPInferenceEngine, MAPInferenceEngineError
from prophetverse.engine.optimizer import AdamOptimizer


def test_raises_error_when_nan_loss():

    bad_svi_result = SVIRunResult(
        params={"param1": jnp.array([1, 2, 3])},
        state=None,
        losses=jnp.array([1, 2, jnp.nan]),
    )

    good_svi_result = SVIRunResult(
        params={"param1": jnp.array([1, 2, 3])}, state=None, losses=jnp.array([1, 2, 3])
    )

    inf_engine = MAPInferenceEngine(optimizer=AdamOptimizer())
    assert jnp.array_equal(inf_engine._rng_key, jax.random.PRNGKey(0)) # Default key is 0
    assert inf_engine.raise_error_if_nan_loss(good_svi_result) is None
    with pytest.raises(MAPInferenceEngineError):
        inf_engine.raise_error_if_nan_loss(bad_svi_result)
