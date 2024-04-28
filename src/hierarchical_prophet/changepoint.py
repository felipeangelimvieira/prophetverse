import jax.numpy as jnp

def get_changepoint_matrix(t, changepoint_t) -> jnp.ndarray:

    expanded_ts = jnp.tile(t.reshape((-1, 1)), (1, len(changepoint_t)))
    A = (expanded_ts >= changepoint_t.reshape((1, -1))).astype(int) * expanded_ts
    cutoff_ts = changepoint_t.reshape((1, -1))
    A = jnp.clip(A - cutoff_ts + 1, 0, None)
    return A


def get_changepoint_timeindexes(
    t, changepoint_interval: int, changepoint_range: float = 0.90
) -> jnp.array:
    
    if changepoint_range < 1 and changepoint_range > 0:
        max_t = t.max() * changepoint_range
    elif changepoint_range >= 1:
        max_t = changepoint_range
    else:
        max_t = t.max() + changepoint_range
    
    changepoint_t = jnp.arange(0, max_t, changepoint_interval)
    return changepoint_t
