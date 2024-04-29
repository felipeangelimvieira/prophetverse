import pytest
import numpy as np
import jax.numpy as jnp
from prophetverse.models.multivariate_model.changepoint import (
    compute_changepoint_design_matrix,
) 


def test_single_series_single_changepoint():
    t = np.arange(10)
    changepoint_ts = [[5]]
    expected = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]).reshape((1, 10, 1))
    result = compute_changepoint_design_matrix(t, changepoint_ts)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series single changepoint",
    )


def test_single_series_multiple_changepoints():
    t = np.arange(7)
    changepoint_ts = [[2, 5]]
    expected = np.array([[0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 1, 2]]).reshape((1, 2, 7)).transpose((0, 2,1))
    result = compute_changepoint_design_matrix(t, changepoint_ts)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series multiple changepoints",
    )


def test_multiple_series():
    t = np.arange(10)
    changepoint_ts = [[5], [3]]
    expected = np.concatenate([
        np.concatenate([np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]).reshape((1, 10, 1)), np.zeros((1, 10, 1))], axis=-1),
        np.concatenate([np.zeros((1, 10, 1)), np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]).reshape((1, 10, 1))], axis=-1),
    ], axis=0)
    
    result = compute_changepoint_design_matrix(t, changepoint_ts)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series single changepoint",
    )



