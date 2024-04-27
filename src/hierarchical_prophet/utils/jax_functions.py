import jax
import jax.numpy as jnp


#@jax.jit
def get_changepoint_offset_adjustment(
    changepoints_matrix, changepoint_t, changepoint_coefficients
):  

    g = (-changepoint_t * changepoint_coefficients).reshape((-1, 1))

    if changepoints_matrix.ndim == 3:
        g = jnp.expand_dims(g, axis=0)
        g = jnp.tile(g, (changepoints_matrix.shape[0], 1, 1))

    return (
        changepoints_matrix
        @ g
    )


#@jax.jit
def get_changepoint_slopes(
    t: jnp.ndarray,
    changepoints_matrix: jnp.ndarray,
    changepoint_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the changepoint slopes.

    Args:
        t: The input array.
        changepoints_matrix: The matrix of changepoints.
        changepoint_coefficients: The coefficients for each changepoint.

    Returns:
        The calculated changepoint slopes.
    """
    changepoint_coefficients = changepoint_coefficients.reshape((-1 ,1))
    if t.ndim == 1:
        t = t.reshape((-1, 1))
    if changepoints_matrix.ndim == 3:
        changepoint_coefficients = jnp.expand_dims(changepoint_coefficients, axis=0)
        changepoint_coefficients = jnp.tile(
            changepoint_coefficients, (changepoints_matrix.shape[0], 1, 1)
        )
    return (changepoints_matrix @ changepoint_coefficients) * t


@jax.jit
def get_changepoint_coefficient_matrix(changepoints_matrix, changepoint_coefficients):
    return (changepoints_matrix) @ changepoint_coefficients.reshape((-1, 1))




@jax.jit
def additive_mean_model(trend, *args):
    mean = trend
    for i in range(len(args)):
        mean += args[i]
    return mean


@jax.jit
def multiplicative_mean_model(trend, *args, exponent=1):
    mean = trend
    
    for i in range(len(args)):
        effect = (trend**exponent)*(args[i])
        mean += effect
    return mean
