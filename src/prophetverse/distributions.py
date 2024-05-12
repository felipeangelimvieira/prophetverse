from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes


class GammaReparametrized(dist.Gamma):
    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale=1.0, *, validate_args=None):

        self.loc, self.scale = promote_shapes(loc, scale)

        rate = loc/ (scale ** 2)
        concentration = loc * rate

        super(GammaReparametrized, self).__init__(
            rate=rate, concentration=concentration, validate_args=validate_args
        )
