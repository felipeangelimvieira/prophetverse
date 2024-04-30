import pytest
import pandas as pd
import numpy as np
from jax import numpy as jnp
from prophetverse.models.multivariate_model.multiindex import (
    get_bottom_series_idx,
    get_multiindex_loc,
    loc_bottom_series,
    iterate_all_series,
    convert_index_to_days_since_epoch,
    series_to_tensor,
    extract_timetensor_from_dataframe,
    convert_dataframe_to_tensors,
    reindex_time_series
)


# Sample data preparation
@pytest.fixture
def sample_hierarchical_data():
    levels = [("A", i) for i in range(3)] + [("B", i) for i in range(3)]
    idx = pd.MultiIndex.from_tuples(levels, names=["Level1", "Level2"])
    data = np.random.randn(6, 2)
    return pd.DataFrame(data, index=idx, columns=["Feature1", "Feature2"])


@pytest.fixture
def sample_time_index():
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    return pd.PeriodIndex(dates, freq="D")


# Test for getting bottom series index
def test_get_bottom_series_idx(sample_hierarchical_data):
    idx = get_bottom_series_idx(sample_hierarchical_data)
    assert idx.equals(pd.Index(["A", "B"]))


# Test for locating data using multiindex
def test_get_multiindex_loc(sample_hierarchical_data):
    result = get_multiindex_loc(sample_hierarchical_data, [("A", 1)])
    assert (
        not result.empty and len(result) == 1
    ), "Should return 1 rows matching the multi-index ('A', 1)"


# Test for fetching bottom series
def test_loc_bottom_series(sample_hierarchical_data):
    result = loc_bottom_series(sample_hierarchical_data)
    assert isinstance(
        result, pd.DataFrame
    ), "Should return a DataFrame of the bottom series"


# Test for iterating all series
def test_iterate_all_series(sample_hierarchical_data):
    for idx, series in iterate_all_series(sample_hierarchical_data):
        assert len(series) == 3, "Each iterated series should have 2 features"


# Test for converting index to days since epoch
def test_convert_index_to_days_since_epoch(sample_time_index):
    result = convert_index_to_days_since_epoch(sample_time_index)
    assert result.shape == (10,), "Should convert 10 dates into days since epoch"


# Test for converting a DataFrame series to a JAX tensor
def test_series_to_tensor(sample_hierarchical_data):
    result = series_to_tensor(sample_hierarchical_data)
    assert isinstance(result, jnp.ndarray) and result.shape == (
        2,
        3,
        2,
    ), "Shape should reflect the reshaping into a 3D tensor"



# Test for converting DataFrame to tensors
def test_convert_dataframe_to_tensors(sample_hierarchical_data):
    t_arrays, df_as_arrays = convert_dataframe_to_tensors(sample_hierarchical_data)
    assert isinstance(t_arrays, jnp.ndarray) and isinstance(
        df_as_arrays, jnp.ndarray
    ), "Should convert both time and data arrays to JAX tensors"


def test_reindex_timeseries():

    df = pd.DataFrame(
        data=np.random.randn(10, 2),
        index=pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range(start="2021-01-01", periods=5, freq="D")],
            names=["group", "time"],
        ),
        columns=["value1", "value2"],
    )

    output = reindex_time_series(
        df, pd.date_range(start="2021-01-01", periods=10, freq="D")
    )

    assert output.index.equals(
        pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range(start="2021-01-01", periods=10, freq="D")],
            names=["group", "time"],
        )
    )
