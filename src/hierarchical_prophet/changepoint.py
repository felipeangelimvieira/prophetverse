import jax.numpy as jnp

def get_changepoint_matrix(t : jnp.ndarray, changepoint_t : jnp.array) -> jnp.ndarray:
    """Generates a changepoint matrix based on the time indexes and changepoint time indexes.

    Args:
        t (jnp.ndarray): array with timepoints of shape (n, 1) preferably
        changepoint_t (jnp.array): array with changepoint timepoints of shape (n_changepoints,)

    Returns:
        jnp.ndarray: changepoint matrix - already with discontinuities at changepoits hanlded
    """
    

    expanded_ts = jnp.tile(t.reshape((-1, 1)), (1, len(changepoint_t)))
    A = (expanded_ts >= changepoint_t.reshape((1, -1))).astype(int) * expanded_ts
    cutoff_ts = changepoint_t.reshape((1, -1))
    A = jnp.clip(A - cutoff_ts + 1, 0, None)
    return A


def get_changepoint_timeindexes(
    t: jnp.ndarray, changepoint_interval: int, changepoint_range: float = 0.90
) -> jnp.array:
    """
    Returns an array of time indexes for changepoints based on the given parameters.

    Args:
        t (jnp.ndarray): The array of time values.
        changepoint_interval (int): The interval between changepoints.
        changepoint_range (float, optional): The range of changepoints. Defaults to 0.90.
            If greater than 1, it is interpreted as then number of timepoints.
            If less than zero, it is interpreted as number of timepoints from the end of the time series.

    Returns:
        jnp.array: An array of time indexes for changepoints.
    """
    if changepoint_range < 1 and changepoint_range > 0:
        max_t = t.max() * changepoint_range
    elif changepoint_range >= 1:
        max_t = changepoint_range
    else:
        max_t = t.max() + changepoint_range
    
    changepoint_t = jnp.arange(0, max_t, changepoint_interval)
    return changepoint_t
