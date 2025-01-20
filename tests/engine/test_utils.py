import numpy as np
import numpyro
import pytest
from jax.random import PRNGKey
from numpyro.distributions import Normal, LKJCholesky
from numpyro.diagnostics import summary
from numpyro.infer import NUTS, MCMC

from prophetverse.engine.utils import assert_mcmc_converged
from prophetverse.exc import ConvergenceError


@pytest.fixture
def rng_key():
    return PRNGKey(123)


@pytest.fixture
def shape():
    return 4, 500


@pytest.fixture
def converged_summary(rng_key, shape):
    samples = {
        "alpha": Normal().sample(rng_key, shape),
        "beta": LKJCholesky(4).sample(rng_key, shape),
        "sigma": Normal().expand((2, 2)).to_event(2).sample(rng_key, shape)
    }

    return summary(samples, group_by_chain=len(shape) > 1)


@pytest.fixture
def not_converged_summary(rng_key, shape):
    def model():
        numpyro.sample("beta", LKJCholesky(4))
        return


    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=5, num_samples=5, num_chains=4, progress_bar=False)
    mcmc.run(rng_key)

    samples = mcmc.get_samples(group_by_chain=True)

    return summary(samples, group_by_chain=True)


def test_assert_converged(converged_summary):
    assert_mcmc_converged(converged_summary, max_r_hat=1.1)


def test_assert_error(not_converged_summary):
    with pytest.raises(ConvergenceError):
        assert_mcmc_converged(not_converged_summary, max_r_hat=1.1)
