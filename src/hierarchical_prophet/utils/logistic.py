import numpy as np


def suggest_logistic_rate_and_offset(
    t: np.ndarray, y: np.ndarray, capacities: float or np.ndarray
):
    """
    Suggests the logistic rate and offset based on the given time series data.

    Parameters:
        t (ndarray): The time values of the time series data.
        y (ndarray): The observed values of the time series data.
        capacities (float or ndarray): The capacity or capacities of the time series data.

    Returns:
        m (ndarray): The suggested offset.
        k (ndarray): The suggested logistic rate.

    """

    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim == 3:
        # Shape here would be (n_series, n_samples, 1)
        y = y.squeeze()
    if t.ndim == 1:
        t = t.reshape(1, -1)
    elif t.ndim == 3:
        # Shape here would be (n_series, n_samples, 1)
        t = t.squeeze()

    i0, i1 = t.argmin(axis=1), t.argmax(axis=1)
    t0, t1 = t[:, i0].flatten(), t[:, i1].flatten()
    T = t0 - t1
    y0, y1 = y[:, i0].flatten(), y[:, i1].flatten()

    r0 = capacities / y0
    r1 = capacities / y1

    L0 = np.log(r0 - 1)
    L1 = np.log(r1 - 1)

    
    k = (L1 - L0) / T
    m = - (L1 + k * t1)

    return k, m
