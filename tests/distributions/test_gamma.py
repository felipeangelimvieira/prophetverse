import pytest
from jax import random
from numpyro import distributions as dist

from prophetverse.distributions import GammaReparametrized


@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 1.0),
        (10.0, 1.0),
        (5.0, 2.0),
    ],
)
def test_gamma_reparametrized_init(loc, scale):
    dist_test = GammaReparametrized(loc, scale)
    assert dist_test.loc == loc
    assert dist_test.scale == scale
    assert isinstance(dist_test, GammaReparametrized)


@pytest.mark.parametrize(
    "loc, scale",
    [
        (1.0, 1.0),
        (10.0, 1.0),
        (5.0, 2.0),
    ],
)
def test_gamma_reparametrized_moments(loc, scale):
    dist_test = GammaReparametrized(loc, scale)
    mean_expected = loc
    var_expected = scale**2
    assert dist_test.mean == pytest.approx(mean_expected)
    assert dist_test.variance == pytest.approx(var_expected)


def test_gamma_reparametrized_sample_shape():
    key = random.PRNGKey(0)
    dist_test = GammaReparametrized(10.0, 1.0)
    samples = dist_test.sample(key, (100,))
    assert samples.shape == (100,)


@pytest.mark.parametrize("value", [0.5, 1.5, 3.5])
def test_gamma_reparametrized_log_prob(value):
    loc = 10.0
    scale = 2.0
    dist_test = GammaReparametrized(loc, scale)
    rate = loc / (scale**2)
    concentration = loc * rate
    dist_standard = dist.Gamma(rate=rate, concentration=concentration)
    log_prob_reparam = dist_test.log_prob(value)
    log_prob_standard = dist_standard.log_prob(value)
    assert log_prob_reparam == pytest.approx(log_prob_standard)
