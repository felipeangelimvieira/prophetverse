import jax.numpy as jnp
import pytest
from numpyro.infer.svi import SVIRunResult

from prophetverse.engine import MAPInferenceEngine, MAPInferenceEngineError


def test_raises_error_when_nan_loss():

    bad_svi_result = SVIRunResult(
        params={"param1": jnp.array([1, 2, 3])},
        state=None,
        losses=jnp.array([1, 2, jnp.nan]),
    )

    good_svi_result = SVIRunResult(
        params={"param1": jnp.array([1, 2, 3])}, state=None, losses=jnp.array([1, 2, 3])
    )

    inf_engine = MAPInferenceEngine(lambda *args: None)

    assert inf_engine.raise_error_if_nan_loss(good_svi_result) is None
    with pytest.raises(MAPInferenceEngineError):
        inf_engine.raise_error_if_nan_loss(bad_svi_result)
