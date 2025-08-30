import numpy as np
import numpyro
import pandas as pd
import pytest

from prophetverse.effects.trend.piecewise import (
    PiecewiseLinearTrend,
    PiecewiseLogisticTrend,
    _enforce_array_if_zero_dim,
    _get_changepoint_matrix,
    _get_changepoint_timeindexes,
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
        fh = pd.date_range(start="2020-01-01", periods=100, freq="D")
        data = trend_model.transform(X=df, fh=fh)
        with numpyro.handlers.seed(rng_seed=0):
            trend = trend_model.predict(data, predicted_effects={})
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
