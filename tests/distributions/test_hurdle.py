import pytest
import jax.numpy as jnp
import jax.random as jrnd
from numpyro.distributions import NegativeBinomial2, Poisson

from prophetverse.distributions import TruncatedDiscrete, HurdleDistribution


@pytest.mark.parametrize(
    "distribution",
    [
        ("poisson", {"rate": 1.0}),
        ("negative-binomial", {"mean": 1.0, "concentration": 1.0}),
    ],
)
def test_zero_truncated_distribution(distribution):
    dist_name, params = distribution

    if dist_name == "poisson":
        base_dist = Poisson(**params)
    elif dist_name == "negative-binomial":
        base_dist = NegativeBinomial2(**params)

    new_dist = TruncatedDiscrete(base_dist)
    samples = new_dist.sample(jrnd.key(123), (1_000,))

    bad_log_prob = new_dist.log_prob(0)
    assert bad_log_prob == -float("inf"), "Log probability at zero should be -inf"

    assert (samples.min() > 0).all()
    assert (new_dist.log_prob(samples) > base_dist.log_prob(samples)).all()


@pytest.mark.parametrize(
    "prob_gt_zero, distribution, params, value",
    [
        (0.3, "poisson", {"rate": 1.5}, 1),
        (0.7, "poisson", {"rate": 2.0}, 2),
        (0.4, "negative-binomial", {"mean": 1.2, "concentration": 1.5}, 1),
        (0.6, "negative-binomial", {"mean": 2.5, "concentration": 2.0}, 3),
    ],
)
def test_hurdle_log_prob_positive(prob_gt_zero, distribution, params, value):
    if distribution == "poisson":
        base_dist = Poisson(**params)
    else:
        base_dist = NegativeBinomial2(**params)

    truncated = TruncatedDiscrete(base_dist)
    hurdle = HurdleDistribution(jnp.array(prob_gt_zero), truncated)

    # Expected: log(p) + truncated.log_prob(value)
    expected = jnp.log(prob_gt_zero) + truncated.log_prob(value)
    assert hurdle.log_prob(value) == pytest.approx(float(expected))


@pytest.mark.parametrize("prob_gt_zero", [0.1, 0.3, 0.5, 0.8])
def test_hurdle_log_prob_zero(prob_gt_zero):
    base_dist = Poisson(rate=1.5)
    truncated = TruncatedDiscrete(base_dist)
    hurdle = HurdleDistribution(jnp.array(prob_gt_zero), truncated)

    lp_zero = hurdle.log_prob(0)
    expected = jnp.log1p(-prob_gt_zero)
    assert lp_zero == pytest.approx(float(expected))


@pytest.mark.parametrize(
    "prob_gt_zero, distribution, params",
    [
        (0.2, "poisson", {"rate": 1.0}),
        (0.6, "poisson", {"rate": 2.5}),
        (0.4, "negative-binomial", {"mean": 1.5, "concentration": 1.2}),
        (0.7, "negative-binomial", {"mean": 2.0, "concentration": 2.5}),
    ],
)
def test_hurdle_sample(prob_gt_zero, distribution, params):
    if distribution == "poisson":
        base_dist = Poisson(**params)
    else:
        base_dist = NegativeBinomial2(**params)

    truncated = TruncatedDiscrete(base_dist)
    hurdle = HurdleDistribution(jnp.array(prob_gt_zero), truncated)

    key = jrnd.key(321)
    n = 5_000
    samples = hurdle.sample(key, (n,))

    # Support check
    assert (samples >= 0).all()

    # Empirical zero frequency approximates (1 - p)
    zero_freq = (samples == 0).mean()
    expected_zero = 1.0 - prob_gt_zero
    # Allow a tolerance ~3 standard errors: sqrt(p*(1-p)/n)
    se = (expected_zero * (1 - expected_zero) / n) ** 0.5
    assert abs(float(zero_freq) - expected_zero) < 3.5 * se + 0.01  # minimum slack

    # Positive samples (if any) should be > 0
    positives = samples[samples > 0]
    if positives.size > 0:
        assert (positives > 0).all()
