import numpy as np
import pytest
from prophetverse.sktime.multivariate import (
    TimeScaler,
) 


def test_fit_with_single_difference():
    """
    Test the fit method with a time series where the difference between each timestamp is consistent.
    """
    t = np.array([[0, 1, 2, 3]])
    scaler = TimeScaler().fit(t)

    assert scaler.t_scale == 1
    assert scaler.t_min == 0


def test_scale_after_fit():
    """
    Test the scale method on a fitted TimeScaler object.
    """
    t = np.array([2, 3, 4, 5])
    expected_scaled_t = np.array([0, 1, 2, 3])
    scaler = TimeScaler().fit(t)
    scaled_t = scaler.scale(t)

    np.testing.assert_array_equal(scaled_t, expected_scaled_t)


def test_fit_scale_combined():
    """
    Test the fit_scale method, which combines fitting and scaling in a single step.
    """
    t = np.array([[10, 12, 14, 16]])
    expected_scaled_t = np.array([[0, 1, 2, 3]])
    scaled_t = TimeScaler().fit_scale(t)

    np.testing.assert_array_equal(scaled_t, expected_scaled_t)
