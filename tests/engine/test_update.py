import numpy as np
import numpyro
import pytest
from numpyro.distributions import HalfNormal, Normal

from prophetverse import MCMCInferenceEngine


@pytest.fixture
def data():
    return np.random.normal(size=100)


def model(y: np.ndarray):
    mean = numpyro.sample("mean", Normal())
    scale = numpyro.sample("scale", HalfNormal())

    numpyro.sample("obs", Normal(mean, scale), obs=y)

    return


@pytest.mark.parametrize("mode", ["mean"])
def test_update_mcmc(data, mode):
    engine = MCMCInferenceEngine(
        10,
        20,
        progress_bar=False,
    )

    # fit
    engine.infer(model, y=data)
    original_samples = engine.posterior_samples_

    # update
    new_data = 5.0 + np.random.normal(size=100)
    new_data = np.concatenate([data, new_data], axis=0)

    engine.update(["mean"], mode=mode, y=new_data)

    new_samples = engine.posterior_samples_

    assert new_samples["mean"].mean() > original_samples["mean"].mean()
    assert np.allclose(new_samples["scale"].mean(), original_samples["scale"].mean())

    return
