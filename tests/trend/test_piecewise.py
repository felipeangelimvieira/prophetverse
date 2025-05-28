import numpy as np
import numpyro
import pandas as pd
import pytest

from prophetverse.effects.trend.piecewise import (
    PiecewiseLinearTrend,
    PiecewiseFlatTrend,
    PiecewiseLogisticTrend,
    _enforce_array_if_zero_dim,
    _get_changepoint_matrix,
    _get_changepoint_timeindexes,
    _get_flat_changepoint_matrix,
    _suggest_logistic_rate_and_offset,
    _to_list_if_scalar,
)


def _make_mock_dataframe():
    return pd.DataFrame(
        {
            "value": np.random.rand(100),
        },
        index=pd.date_range("20200101", periods=100, freq="D"),
    )


def _make_mock_multiindex_dataframe():
    dates = pd.date_range("20200101", periods=100, freq="D")
    series = ["series1", "series2"]
    index = pd.MultiIndex.from_product([series, dates], names=["-2", "-1"])
    values = np.random.rand(len(index))
    return pd.DataFrame({"value": values}, index=index)


@pytest.fixture
def mock_dataframe():
    return _make_mock_dataframe()


# Fixtures for common test objects
@pytest.fixture
def mock_multiindex_dataframe():
    return _make_mock_multiindex_dataframe()


@pytest.fixture
def piecewise_linear_trend():
    return PiecewiseLinearTrend(
        changepoint_interval=10,
        changepoint_range=90,
        changepoint_prior_scale=1,
    )


@pytest.fixture
def piecewise_logistic_trend():
    return PiecewiseLogisticTrend(
        changepoint_interval=10, changepoint_range=90, changepoint_prior_scale=0.5
    )


@pytest.fixture
def piecewise_flat_trend():
    return PiecewiseFlatTrend(
        changepoint_interval=10,
        changepoint_range=90,
        changepoint_prior_scale=0.001,
    )


# Tests for PiecewiseLinearTrend
def test_piecewise_linear_initialize(piecewise_linear_trend, mock_dataframe):
    piecewise_linear_trend.fit(mock_dataframe, mock_dataframe)
    assert hasattr(
        piecewise_linear_trend, "_changepoint_ts"
    ), "Changepoint ts not set during initialization."


# Tests for PiecewiseLogisticTrend
def test_piecewise_logistic_initialize(piecewise_logistic_trend, mock_dataframe):
    piecewise_logistic_trend.fit(mock_dataframe, mock_dataframe)
    piecewise_logistic_trend.transform(
        mock_dataframe, fh=mock_dataframe.index.get_level_values(-1)
    )
    assert hasattr(
        piecewise_logistic_trend, "_changepoint_ts"
    ), "Changepoint ts not set during initialization."


# Tests for PiecewiseFlatTrend
def test_piecewise_flat_initialize(piecewise_flat_trend, mock_dataframe):
    """Test that PiecewiseFlatTrend initializes properly and sets changepoint attributes."""
    piecewise_flat_trend.fit(mock_dataframe, mock_dataframe)
    assert hasattr(
        piecewise_flat_trend, "_changepoint_ts"
    ), "Changepoint ts not set during initialization."

    # Test that it inherited attributes from PiecewiseLinearTrend
    assert hasattr(piecewise_flat_trend, "changepoint_interval")
    assert hasattr(piecewise_flat_trend, "changepoint_range")
    assert hasattr(piecewise_flat_trend, "changepoint_prior_scale")


def test_piecewise_flat_fit_and_transform(piecewise_flat_trend, mock_dataframe):
    """Test PiecewiseFlatTrend fit and transform methods."""
    # Fit the model
    piecewise_flat_trend.fit(mock_dataframe, mock_dataframe)

    # Test transformation
    result = piecewise_flat_trend.transform(
        mock_dataframe, fh=mock_dataframe.index.get_level_values(-1)
    )

    # Should have proper shape (n_samples, n_changepoints)
    assert result.shape[0] == len(mock_dataframe)
    assert result.shape[1] > 0  # Should have at least one changepoint


def test_piecewise_flat_get_changepoint_matrix(piecewise_flat_trend, mock_dataframe):
    """Test that flat trend generates changepoint matrix correctly."""
    piecewise_flat_trend.fit(mock_dataframe, mock_dataframe)
    period_index = pd.period_range(start="2020-01-01", periods=100, freq="D")
    result = piecewise_flat_trend.get_changepoint_matrix(period_index)

    # Should return proper matrix shape
    assert result.shape == (100, 9), "Changepoint matrix shape is incorrect."

    # Matrix values should be 0 or 1 for flat trend (step function)
    unique_values = np.unique(result)
    assert all(val in [0.0, 1.0] for val in unique_values), "Flat trend matrix should only contain 0s and 1s"


def test_piecewise_flat_predict(piecewise_flat_trend, mock_dataframe):
    """Test PiecewiseFlatTrend prediction functionality."""
    piecewise_flat_trend.fit(mock_dataframe, mock_dataframe)
    period_index = pd.period_range(start="2020-01-01", periods=100, freq="D")
    changepoint_matrix = piecewise_flat_trend.get_changepoint_matrix(period_index)

    with numpyro.handlers.seed(rng_seed=42):
        trend = piecewise_flat_trend.predict(changepoint_matrix, predicted_effects={})

    # Check output shape and properties
    assert trend.ndim == 2
    assert trend.shape == (100, 1)
    assert not np.isnan(trend).any(), "Prediction should not contain NaN values"


def test_piecewise_flat_multivariate(piecewise_flat_trend, mock_multiindex_dataframe):
    """Test PiecewiseFlatTrend with multivariate data."""
    piecewise_flat_trend.fit(mock_multiindex_dataframe, mock_multiindex_dataframe)

    # Test transformation with multivariate data
    result = piecewise_flat_trend.transform(
        mock_multiindex_dataframe,
        fh=mock_multiindex_dataframe.index.get_level_values(-1)
    )

    # Test prediction with multivariate
    with numpyro.handlers.seed(rng_seed=42):
        trend = piecewise_flat_trend.predict(result, predicted_effects={})

    assert trend.ndim == 3  # Should be 3D for multivariate
    assert trend.shape[-1] == 1
    assert not np.isnan(trend).any()


def test_get_flat_changepoint_matrix():
    """Test the _get_flat_changepoint_matrix utility function directly."""
    t = np.arange(10)
    changepoint_ts = np.array([[3, 7]])

    # Get flat changepoint matrix
    flat_matrix = _get_flat_changepoint_matrix(t, changepoint_ts)

    # Should be step functions (0s and 1s only)
    expected_first_cp = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # Step at t=3
    expected_second_cp = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # Step at t=7

    expected = np.column_stack([expected_first_cp, expected_second_cp])

    print(expected)
    print(flat_matrix)
    np.testing.assert_array_equal(
        flat_matrix,
        expected,
        "Flat changepoint matrix should create step functions"
    )


def test_flat_vs_linear_changepoint_matrix_difference():
    """Test that flat and linear changepoint matrices are different."""
    t = np.arange(10)
    changepoint_ts = np.array([[3, 7]])

    # Get both matrices
    flat_matrix = _get_flat_changepoint_matrix(t, changepoint_ts)
    linear_matrix = _get_changepoint_matrix(t, changepoint_ts)

    # They should be different (linear has ramps, flat has steps)
    assert not np.array_equal(flat_matrix, linear_matrix), \
        "Flat and linear changepoint matrices should be different"

    # Flat matrix should only have 0s and 1s
    flat_unique = np.unique(flat_matrix)
    assert all(val in [0.0, 1.0] for val in flat_unique), \
        "Flat matrix should only contain 0s and 1s"

    # Linear matrix should have more varied values (ramps)
    linear_unique = np.unique(linear_matrix)
    assert len(linear_unique) > 2, \
        "Linear matrix should have more than just 0s and 1s"


def test_piecewise_flat_changepoint_prior_vector(piecewise_flat_trend, mock_dataframe):
    """Test that PiecewiseFlatTrend properly handles changepoint prior scale vector."""
    piecewise_flat_trend.fit(mock_dataframe, mock_dataframe)

    # Should have proper number of changepoints
    assert hasattr(piecewise_flat_trend, "_changepoint_ts")
    n_changepoints = len(piecewise_flat_trend._changepoint_ts[0])

    # Test with scalar prior scale (should be converted to vector)
    assert piecewise_flat_trend.changepoint_prior_scale == 0.001

    # Test with vector prior scale
    flat_trend_vector = PiecewiseFlatTrend(
        changepoint_interval=10,
        changepoint_range=90,
        changepoint_prior_scale=[0.001, 0.002, 0.001, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001]
    )
    flat_trend_vector.fit(mock_dataframe, mock_dataframe)

    # Should handle vector input properly
    assert len(flat_trend_vector.changepoint_prior_scale) == n_changepoints


def test_piecewise_flat_suggest_offset():
    """Test the _suggest_offset method for PiecewiseFlatTrend."""
    flat_trend = PiecewiseFlatTrend(
        changepoint_interval=10,
        changepoint_range=90,
        changepoint_prior_scale=0.001
    )

    # Create simple test data
    t = np.arange(20)
    y = pd.DataFrame({"y": np.ones(20) * 5.0})  # Constant value

    offset = flat_trend._suggest_offset(y)

    # For constant data, offset should be close to the mean
    assert abs(offset - 5.0) < 1e-6, "Offset should match the constant value"

    # Test with trending data
    y_trend = t * 0.5 + 3.0  # Linear trend
    y_trend = pd.DataFrame({"y" : y_trend})
    offset_trend = flat_trend._suggest_offset(y_trend)

    # Should suggest reasonable offset
    assert isinstance(offset_trend.item(), (int, float, np.number)), "Offset should be numeric"

@pytest.mark.parametrize(
    "make_df,expected_ndim",
    [(_make_mock_dataframe, 2), (_make_mock_multiindex_dataframe, 3)],
)
def test_piecewise_compute_trend(
    piecewise_linear_trend, piecewise_logistic_trend, make_df, expected_ndim
):
    df = make_df()

    for trend_model in [piecewise_linear_trend, piecewise_logistic_trend]:
        trend_model.fit(df, df)
        period_index = pd.period_range(start="2020-01-01", periods=100, freq="D")
        changepoint_matrix = trend_model.get_changepoint_matrix(period_index)
        with numpyro.handlers.seed(rng_seed=0):
            trend = trend_model.predict(changepoint_matrix, predicted_effects={})
        assert (
            trend.ndim == expected_ndim
        ), f"Dimensions are incorrect for trend_model {trend_model.__class__.__name__}"
        assert trend.shape[-1] == 1
        assert trend.shape[-2] == 100


# Tests for utility functions
def test_to_list_if_scalar():
    assert _to_list_if_scalar(5, size=3) == [
        5,
        5,
        5,
    ], "Scalar to list conversion failed."
    assert _to_list_if_scalar([5, 6], size=3) == [5, 6], "List input should not change."


def test_enforce_array_if_zero_dim():
    x = np.array(5)
    result = _enforce_array_if_zero_dim(x)
    assert result.shape == (1,), "Dimension enforcement failed."


def test_get_changepoint_timeindexes():
    t = np.linspace(0, 100, 101)
    changepoints = _get_changepoint_timeindexes(t, 10, 50)
    assert len(changepoints) > 0, "Changepoint time indexes calculation failed."


def test_piecewise_linear_get_changepoint_matrix(
    piecewise_linear_trend, mock_dataframe
):
    piecewise_linear_trend.fit(mock_dataframe, mock_dataframe)
    period_index = pd.period_range(start="2020-01-01", periods=100, freq="D")
    result = piecewise_linear_trend.get_changepoint_matrix(period_index)

    assert result.shape == (100, 9), "Changepoint matrix shape is incorrect."


def test_single_series_get_multivariate_changepoint_matrix(piecewise_linear_trend):
    t = np.arange(7)
    changepoint_ts = np.array([[2, 5]])
    expected = (
        np.array([[0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 0, 0, 1]])
        .reshape((1, 2, 7))
        .transpose((0, 2, 1))
    )

    piecewise_linear_trend._changepoint_ts = changepoint_ts
    result = piecewise_linear_trend._get_multivariate_changepoint_matrix(t)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series multiple changepoints",
    )


def test_multiple_series_get_multivariate_changepoint_matrix(piecewise_linear_trend):
    t = np.arange(10)
    changepoint_ts = [[5], [3]]
    expected = np.concatenate(
        [
            np.concatenate(
                [
                    np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4]).reshape((1, 10, 1)),
                    np.zeros((1, 10, 1)),
                ],
                axis=-1,
            ),
            np.concatenate(
                [
                    np.zeros((1, 10, 1)),
                    np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6]).reshape((1, 10, 1)),
                ],
                axis=-1,
            ),
        ],
        axis=0,
    )

    piecewise_linear_trend._changepoint_ts = changepoint_ts
    result = piecewise_linear_trend._get_multivariate_changepoint_matrix(t)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series single changepoint",
    )


def test_get_changepoint_matrix():
    t = np.arange(10)
    changepoint_ts = np.array([[5]])
    expected = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4]).reshape((10, 1))
    result = _get_changepoint_matrix(t, changepoint_ts)
    np.testing.assert_array_equal(
        result,
        expected,
        "Matrix does not match expected for single series single changepoint",
    )


def test_suggest_logistic_rate_and_offset():
    t = np.array([0, 1, 2, 3, 4, 5, 3])
    y = 1 / (1 + np.exp(-(t / 10 - 3)))
    capacities = 100
    k, m = _suggest_logistic_rate_and_offset(t, y, capacities)
    np.allclose(m, 3.0)
    np.allclose(k, 0.1)


def test_suggest_logistic_rate_and_offset_raises_error_with_nan():
    t = np.array([1, 2, 3, 4, 5, 3])
    y = np.array([1, 2, 3, 4, 5, np.nan])
    capacities = 100
    with pytest.raises(ValueError):
        _suggest_logistic_rate_and_offset(t, y * np.nan, capacities)
