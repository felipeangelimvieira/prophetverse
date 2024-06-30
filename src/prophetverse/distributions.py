"""
Custom numpyro distributions for the ProphetVerse package.

This module contains custom distributions that can be used as likelihoods or priors
for the models in the ProphetVerse package.
"""

from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes


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
