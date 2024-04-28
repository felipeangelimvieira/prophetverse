import numpy as np
import jax.numpy as jnp

def compute_changepoint_design_matrix(t, changepoint_ts):
    """
    Compute the changepoint design matrix.

    Parameters:
        t (ndarray): Time indices for each series.
        changepoint_ts (list): List of changepoint time indices for each series.

    Returns:
        ndarray: Changepoint design matrix.
    """
    t = np.array(t)
    changepoint_ts = [np.array(x) for x in changepoint_ts]

    n_changepoint_per_series = [len(x) for x in changepoint_ts]
    changepoint_ts = np.concatenate(changepoint_ts)
    changepoint_design_tensor = []
    changepoint_mask_tensor = []
    for i, n_changepoints in enumerate(n_changepoint_per_series):
        expanded_ts = jnp.tile(t.reshape((-1, 1)), (1, sum(n_changepoint_per_series)))
        A = (expanded_ts >= changepoint_ts.reshape((1, -1))).astype(int) * expanded_ts
        cutoff_ts = (
            (expanded_ts < changepoint_ts.reshape((1, -1))).astype(int) * expanded_ts
        ).max(axis=0)
        A = np.clip(A - cutoff_ts, 0, None)

        start_idx = sum(n_changepoint_per_series[:i])
        end_idx = start_idx + n_changepoints
        mask = np.zeros_like(A)
        mask[:, start_idx:end_idx] = 1

        changepoint_design_tensor.append(A)
        changepoint_mask_tensor.append(mask)

    changepoint_design_tensor = np.stack(changepoint_design_tensor, axis=0)
    changepoint_mask_tensor = np.stack(changepoint_mask_tensor, axis=0)
    return jnp.array(changepoint_design_tensor) * jnp.array(changepoint_mask_tensor)
