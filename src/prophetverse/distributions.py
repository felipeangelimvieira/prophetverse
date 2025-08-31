"""
Custom numpyro distributions for the ProphetVerse package.

This module contains custom distributions that can be used as likelihoods or priors
for the models in the ProphetVerse package.
"""

from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes
import jax.numpy as jnp


class GammaReparametrized(dist.Gamma):
    """
    A reparametrized version of the Gamma distribution.

    This distribution is reparametrized in terms of loc and scale instead of rate and
    concentration. This makes it easier to specify priors for the parameters.

    Parameters
    ----------
    loc : float or jnp.ndarray
        The location parameter of the distribution.
    scale : float or jnp.ndarray
        The scale parameter of the distribution.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale=1.0, *, validate_args=None):

        self.loc, self.scale = promote_shapes(loc, scale)

        rate = loc / (scale**2)
        concentration = loc * rate

        super().__init__(
            rate=rate, concentration=concentration, validate_args=validate_args
        )


class BetaReparametrized(dist.Beta):
    """Beta distribution parameterized by mean (``loc``) and variance factor (``factor``).

    So smaller factor -> smaller variance (higher concentration). As factor → 1 the
    variance approaches the Bernoulli upper bound μ(1-μ); as factor → 0 the variance
    shrinks to 0.

    Parameters
    ----------
    loc : float or jnp.ndarray
        Mean μ in (0,1).
    factor : float or jnp.ndarray, default 0.2
        Variance factor in (0,1). Var = μ(1-μ)*factor.
    epsilon : float, optional
        Numerical slack used to keep arguments strictly inside valid domain.
    safe : bool, optional
        If True (default) automatically clamps factor to (epsilon, 1-epsilon).
    validate_args : bool, optional
        If True and safe=False, raises for invalid inputs.
    """

    arg_constraints = {
        "loc": constraints.unit_interval,
        "factor": constraints.positive,
    }
    support = constraints.unit_interval
    reparametrized_params = ["loc", "factor"]

    def __init__(self, loc, factor=0.2, *, validate_args=None):

        self.loc = loc
        self.factor = factor

        var = loc * (1 - loc) / (1 + 1 / factor)
        alpha = loc**2 * ((1 - loc) / var - 1 / loc)
        beta = alpha * (1 / loc - 1)

        super().__init__(
            concentration1=alpha, concentration0=beta, validate_args=validate_args
        )
