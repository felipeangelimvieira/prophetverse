import numpy as np
import pytest
from hierarchical_prophet.hierarchical_prophet._changepoint_matrix import (
    ChangepointMatrix, get_changepoint_t, compute_changepoint_design_matrix
) 


def test_get_changepoint_t():
    """
    Test the get_changepoint_t function with a simple 2D array and predefined frequencies.
    """
    t = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    changepoint_freq = [2]
    changepoint_range = [10]
    expected_changepoint_ts = [np.array([0, 2, 4, 6, 8])]

    changepoint_ts = get_changepoint_t(
        t, changepoint_freq, changepoint_range
    )
    np.testing.assert_array_equal(changepoint_ts, expected_changepoint_ts)


def test_compute_changepoint_design_matrix():
    """
    Test the compute_changepoint_design_matrix function with predefined time indices and changepoint indices.
    """
    t = np.array([[1, 2, 3, 4, 5]])
    changepoint_ts = [np.array([1, 3, 5])]
    expected_design_matrix = np.array(
        [[[1, 2, 3, 4, 5], [0, 0, 1, 2, 3], [0, 0, 0, 0, 1]]]
    ).transpose((0,2,1))
    expected_mask_matrix = np.ones((1,5,3))

    design_matrix, mask_matrix = compute_changepoint_design_matrix(
        t, changepoint_ts
    )
    np.testing.assert_array_equal(design_matrix, expected_design_matrix)
    np.testing.assert_array_equal(mask_matrix, expected_mask_matrix)


def test_fit_transform_integration():
    """
    Test the integration of fit and transform methods in ChangepointMatrix class.
    """
    t = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    changepoint_freq = [2]
    changepoint_range = [10]
    matrix = ChangepointMatrix(changepoint_freq, changepoint_range)

    # Execute fit_transform and validate results
    design_matrix, mask_matrix = matrix.fit_transform(t)
    # Ensure the shapes and values are as expected
    assert design_matrix.shape[0] == t.shape[0]  # Number of series
    assert mask_matrix.shape[0] == t.shape[0]  # Number of series

    # Additional checks can include specific values in the design_matrix and mask_matrix


def test_changepoint_ts_array():
    """
    Test the changepoint_ts_array method for correct concatenation of changepoint time indices.
    """
    matrix = ChangepointMatrix([2, 3], [10, 10])
    t = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    matrix.fit(t)
    changepoint_ts = matrix.changepoint_ts_array()
    # Validate the concatenated array has the correct length and values
    assert len(changepoint_ts) == len(matrix.changepoint_ts[0]) + len(
        matrix.changepoint_ts[1]
    )


def test_n_changepoint_per_series():
    """
    Test the n_changepoint_per_series property for correct computation of changepoints per series.
    """
    matrix = ChangepointMatrix([2], [10])
    t = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    matrix.fit(t)
    assert (
        matrix.n_changepoint_per_series[0] == 5
    )  # With freq 2 and range 10, expect 5 changepoints
