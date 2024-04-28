from hierarchical_prophet.utils.logistic import (
    suggest_logistic_rate_and_offset,
    suggest_logistic_rate_and_offset,
)
import numpy as np


def test_suggest_logistic_rate_and_offset():
    # Test case 1: Single capacity, single time series
    t = np.array([1, 2, 3, 4, 5, 3])
    y = 1 / (1 + np.exp(-(t - 3) / 10))
    capacities = 100
    k, m = suggest_logistic_rate_and_offset(t, y, capacities)
    np.allclose(m, 3.0)
    np.allclose(k, 0.1)


def test_suggest_logistic_rate_and_offset():
    # Test case 1: Single capacity, single time series
    t = np.array([1, 2, 3, 4, 5, 3])
    y = 1 / (1 + np.exp(-(t/ 10 - 3)))
    capacities = 100
    k, m = suggest_logistic_rate_and_offset(t, y, capacities)
    np.allclose(m, 3.0)
    np.allclose(k, 0.1)
