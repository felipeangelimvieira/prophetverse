#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict, Tuple, Callable
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist


#
# Protocols for type hinting
#
class EffectFunc(Protocol):
    """Protocol for effect functions. The functions should receive the trend, data and coefficients. The coefficients can be tuples of parameters."""

    def __call__(
        self, trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
    ) -> jnp.ndarray: ...


class SampleParamsFunc(Protocol):

    def __call__(self, **kwargs) -> jnp.ndarray: ...


# --------------
#     Effects
# --------------


# Simple additive and multiplicative effects
# ------------------------------------------


def matrix_multiplication(data, coefficients):
    return data @ coefficients.reshape((-1, 1))


def additive_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:
    return matrix_multiplication(data, coefficients)


def multiplicative_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:
    return trend * matrix_multiplication(data, coefficients)


# Hill function
# -------------


def _apply_exponent_safe(
    data: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:
    """Applies an exponent to given data in a gradient safe way.

    More info on the double jnp.where can be found:
    https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    Args:
      data: Input data to use.
      exponent: Exponent required for the operations.

    Returns:
      The result of the exponent operation with the inputs provided.
    """
    exponent_safe = jnp.where(data == 0, 1, data) ** exponent
    return jnp.where(data == 0, 0, exponent_safe)


def hill(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Calculates the hill function for a given array of values.

    Refer to the following link for detailed information on this equation:
      https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

    Args:
      data: Input data.
      half_max_effective_concentration: ec50 value for the hill function.
      slope: Slope of the hill function.

    Returns:
      The hill values for the respective input data.
    """
    half_max_effective_concentration, slope = coefficients
    save_transform = _apply_exponent_safe(
        data=data / half_max_effective_concentration, exponent=-slope
    )
    return jnp.where(save_transform == 0, 0, 1.0 / (1 + save_transform))


def sample_hill_params(
    *args,
    half_max_concentration: float = 1,
    half_max_rate: float = 1,
    slope_concentration: float = 1,
    slope_rate: float = 1,
    **kwargs
):

    half_max_effective_concentration = numpyro.sample(
        "half_max_effective_concentration",
        dist.Gamma(concentration=half_max_concentration, rate=half_max_rate),
    )

    slope = numpyro.sample(
        "slope", dist.Gamma(concentration=slope_concentration, rate=slope_rate)
    )

    return (half_max_effective_concentration, slope)


# Log effect
# ----------


def log_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:

    scale, rate = coefficients
    return scale * jnp.log(rate * data + 1)


def sample_log_params(
    *args,
    scale_concentration: float = 1,
    scale_rate: float = 1,
    slope_concentration: float = 1,
    slope_rate: float = 1,
    **kwargs,
) -> Tuple[float, float]:

    scale = numpyro.sample(
        "scale", dist.Gamma(concentration=scale_concentration, rate=scale_rate)
    )

    slope = numpyro.sample(
        "slope", dist.Gamma(concentration=slope_concentration, rate=slope_rate)
    )

    return (scale, slope)
