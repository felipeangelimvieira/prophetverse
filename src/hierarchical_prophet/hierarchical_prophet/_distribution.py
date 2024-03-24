import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import ndtr, ndtri
from jax.scipy.stats import norm as jax_norm
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


class NormalReconciled(dist.Distribution):
    """

    A Normal distribution applied to a matrix that projects into a  higher dimensional hyperplane.

    Let $X \sim \mathcal{N}(0_m, 1_m)$. Then, $Y = SX$ where $S$ is a $n \times m$ matrix, is a $n$-dimensional random vector

    """

    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}
    support = dist.constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(
        self, loc=0.0, scale=1.0, reconc_matrix=jnp.eye(1), *, validate_args=None
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))

        loc = loc[..., jnp.newaxis]

        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(reconc_matrix)[:-2]
        )

        self.input_event_shape = jnp.shape(reconc_matrix)[-1:]
        self.loc = loc[..., 0]
        self.scale = jnp.broadcast_to(scale, self.loc.shape)

        self.reconc_matrix = reconc_matrix
        
        self.output_event_shape = (self.reconc_matrix.shape[-2],)

        super(NormalReconciled, self).__init__(
            batch_shape=batch_shape,
            event_shape=self.output_event_shape,
            validate_args=validate_args,
        )

        self.new_scales = (
            jnp.sqrt(self.reconc_matrix**2 @ (self.scale_vector**2))
            .flatten()
            .reshape(self.scale.shape[:-1] + self.output_event_shape)
        )
        self.new_locs = (self.reconc_matrix @ jnp.expand_dims(self.loc, axis=-1))[
            ..., 0
        ]

    @property
    def scale_vector(self):
        return self.scale.reshape((self.scale.shape[0], -1, 1))

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.input_event_shape
        )
        internal_samples = self.loc + eps * self.scale

        return (
            self.reconc_matrix @ jnp.expand_dims(internal_samples, axis=-1)
        ).reshape((-1, *self.output_event_shape))

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.new_scales)
        value_scaled = (value - self.new_locs) / self.new_scales
        return (-0.5 * value_scaled**2 - normalize_term).sum(axis=-1)

    def cdf(self, value):
        scaled = (value - self.new_locs) / self.new_scales
        return ndtr(scaled)

    def log_cdf(self, value):
        return jax_norm.logcdf(value, loc=self.new_locs, scale=self.new_scales)

    def icdf(self, q):
        return self.new_locs + self.new_scales * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.new_locs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.new_scales**2, self.batch_shape)
