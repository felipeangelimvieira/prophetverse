"""Test the sktime contract for Prophet and HierarchicalProphet."""

import pandas as pd
import numpy as np
import pytest  # noqa: F401
from datetime import timedelta
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks
from prophetverse.sktime.event_dummies import EventsDummyTransformer


@parametrize_with_checks(EventsDummyTransformer)
def test_sktime_api_compliance(obj, test_name):
    """Test the sktime contract for Prophet and HierarchicalProphet."""
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


@pytest.fixture
def sample_events():
    # A sample events DataFrame in fbprophet-like format
    df = pd.DataFrame(
        {
            "event_name": ["holiday1", "holiday2"],
            "ds": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")],
            "lower_window": [-1, 0],
            "upper_window": [1, 2],
        }
    )
    return df


@pytest.fixture
def transformer_default(sample_events):
    # Create an instance with dummy_by_window False
    return EventsDummyTransformer(
        events_df=sample_events, prefix="test", dummy_by_window=False
    )


@pytest.fixture
def transformer_window(sample_events):
    # Create an instance with dummy_by_window True
    return EventsDummyTransformer(
        events_df=sample_events, prefix="test", dummy_by_window=True
    )


def test_fit_sets_events_df_(transformer_default, sample_events):
    # Before fitting, the attribute events_df_ should not exist.
    with pytest.raises(AttributeError):
        _ = transformer_default.events_df_
    # Fit the transformer.
    transformer_default.fit(None)
    # After fitting, events_df_ exists and ds is datetime.
    df_fitted = transformer_default.events_df_
    assert isinstance(df_fitted, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df_fitted["ds"])


def test_transform_full_range(transformer_default):
    # Fit and then transform with X=None.
    transformer_default.fit(None)
    X_trans = transformer_default.transform(None)
    # The effective dates are computed from each event.
    # For holiday1: ds = 2020-01-01, window from -1 to +1 => dates 2019-12-31, 2020-01-01, 2020-01-02.
    # For holiday2: ds = 2020-01-10, window from 0 to +2 => dates 2020-01-10, 2020-01-11, 2020-01-12.
    # Thus the full range should cover from 2019-12-31 to 2020-01-12.
    expected_index = pd.date_range(start="2019-12-31", end="2020-01-12", freq="D")
    pd.testing.assert_index_equal(X_trans.index, expected_index)
    # Check that expected columns exist.
    expected_cols = {"test_holiday1", "test_holiday2"}
    assert set(X_trans.columns) == expected_cols
    # All values should be 0 or 1.
    assert X_trans.values.max() <= 1
    assert X_trans.values.min() >= 0


def test_transform_left_join(transformer_default):
    # Create a dummy X with a custom date range.
    date_range = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
    X_dummy = pd.DataFrame(
        np.random.randn(len(date_range), 1), index=date_range, columns=["var1"]
    )
    transformer_default.fit(None)
    X_trans = transformer_default.transform(X_dummy)
    # The output index should match X_dummy's index.
    pd.testing.assert_index_equal(X_trans.index, X_dummy.index)
    # And the dummy columns should be present.
    expected_cols = {"test_holiday1", "test_holiday2", "var1"}
    assert set(X_trans.columns) == expected_cols


def test_dummy_by_window_columns(transformer_window):
    # Test that with dummy_by_window=True, column names include the offset information.
    transformer_window.fit(None)
    X_trans = transformer_window.transform(None)
    # For holiday1, offsets -1, 0, 1 => columns like "test_holiday1_offset_-1", etc.
    cols = X_trans.columns.tolist()
    # Check that at least one column contains "_offset_"
    assert any("_offset_" in col for col in cols)


def test_transform_with_period_index(transformer_default):
    # Create a dummy X with a PeriodIndex.
    period_index = pd.period_range(start="2020-01-01", end="2020-01-05", freq="D")
    # Create a dummy DataFrame using this PeriodIndex.
    X_dummy = pd.DataFrame(
        data=np.random.randn(len(period_index), 1), index=period_index, columns=["var1"]
    )
    transformer_default.fit(None)
    X_trans = transformer_default.transform(X_dummy)

    pd.testing.assert_index_equal(X_trans.index, period_index)
    expected_cols = {"test_holiday1", "test_holiday2", "var1"}
    assert set(X_trans.columns) == expected_cols
