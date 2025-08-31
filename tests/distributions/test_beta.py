import pytest
from jax import random
from numpyro import distributions as dist

from prophetverse.distributions import BetaReparametrized


@pytest.mark.parametrize(
    "loc,factor",
    [
        (0.2, 0.5),
        (0.5, 0.5),
        (0.8, 0.5),
        (0.3, 1.0),
        (0.7, 1.0),
        (0.5, 2.0),
    ],
)
def test_beta_reparametrized_init_attributes(loc, factor):
    d = BetaReparametrized(loc, factor)
    assert isinstance(d, BetaReparametrized)
    assert d.loc == loc
    assert d.factor == factor


@pytest.mark.parametrize(
    "loc,factor",
    [
        (0.2, 0.5),
        (0.5, 0.5),
        (0.8, 0.5),
        (0.3, 1.0),
        (0.7, 1.0),
        (0.5, 2.0),
    ],
)
def test_beta_reparametrized_positive_concentrations(loc, factor):
    d = BetaReparametrized(loc, factor)
    alpha = d.concentration1
    beta = d.concentration0
    # A valid Beta distribution must have positive concentration parameters
    assert alpha > 0, "Alpha (concentration1) must be positive"
    assert beta > 0, "Beta (concentration0) must be positive"


@pytest.mark.parametrize(
    "loc,factor",
    [
        (0.2, 0.5),
        (0.5, 0.5),
        (0.8, 0.5),
        (0.3, 1.0),
        (0.7, 1.0),
        (0.5, 2.0),
    ],
)
def test_beta_reparametrized_mean_matches_loc(loc, factor):
    d = BetaReparametrized(loc, factor)
    alpha = d.concentration1
    beta = d.concentration0
    mean = alpha / (alpha + beta)
    assert mean == pytest.approx(loc, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "loc,factor",
    [
        (0.2, 0.5),
        (0.5, 0.5),
        (0.8, 0.5),
        (0.3, 1.0),
        (0.7, 1.0),
        (0.5, 2.0),
    ],
)
def test_beta_reparametrized_variance_bounds(loc, factor):
    d = BetaReparametrized(loc, factor)
    alpha = d.concentration1
    beta = d.concentration0
    var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    # For any valid Beta: 0 < var < loc(1-loc)
    assert var > 0, "Variance must be positive"
    assert var < loc * (1 - loc) - 1e-12, "Variance must be < loc(1-loc)"


def test_beta_reparametrized_sample_shape_and_support():
    key = random.PRNGKey(0)
    d = BetaReparametrized(0.4, 0.5)
    samples = d.sample(key, (1000,))
    assert samples.shape == (1000,)
    # All samples must lie in (0,1)
    assert (samples > 0).all() and (samples < 1).all()


@pytest.mark.parametrize("value", [0.1, 0.3, 0.6, 0.9])
def test_beta_reparametrized_log_prob_matches_manual(value):
    loc = 0.4
    factor = 0.5
    d = BetaReparametrized(loc, factor)
    alpha = d.concentration1
    beta_param = d.concentration0
    # Use a standard Beta with the same inferred concentrations
    ref = dist.Beta(concentration1=alpha, concentration0=beta_param)
    lp1 = d.log_prob(value)
    lp2 = ref.log_prob(value)
    assert lp1 == pytest.approx(lp2)
