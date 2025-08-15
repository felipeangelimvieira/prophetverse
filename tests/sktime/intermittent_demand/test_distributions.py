import pytest

from prophetverse.sktime.intermittent_demand._truncated_discrete import TruncatedDiscrete


@pytest.mark.smoke
@pytest.mark.parametrize(
    "distribution",
    [
        ("poisson", {"rate": 1.0}),
        ("negative-binomial", {"mean": 1.0, "concentration": 1.0}),
    ],
)
def test_zero_truncated_distribution(distribution):
    import jax.random as jrnd
    from numpyro.distributions import NegativeBinomial2, Poisson

    dist_name, params = distribution

    if dist_name == "poisson":
        distribution = Poisson(**params)
    elif dist_name == "negative-binomial":
        distribution = NegativeBinomial2(**params)

    new_dist = TruncatedDiscrete(distribution)
    samples = new_dist.sample(jrnd.key(123), (1_000,))

    bad_log_prob = new_dist.log_prob(0)
    assert bad_log_prob == -float("inf"), "Log probability at zero should be -inf"

    assert (samples.min() > 0).all()
    assert (new_dist.log_prob(samples) > distribution.log_prob(samples)).all()

    return
