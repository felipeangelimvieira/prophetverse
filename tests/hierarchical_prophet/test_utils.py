import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen

from hierarchical_prophet._utils import (convert_dataframe_to_tensors,
                                    convert_index_to_days_since_epoch,
                                    get_bottom_series_idx, get_multiindex_loc,
                                    iterate_all_series, loc_bottom_series,
                                    series_to_tensor, set_exogenous_priors)

NUM_LEVELS = 2
NUM_BOTTOM_NODES = 3


@pytest.fixture
def multiindex_df():
    agg = Aggregator()
    y = _bottom_hier_datagen(
        no_bottom_nodes=NUM_BOTTOM_NODES,
        no_levels=NUM_LEVELS,
        random_seed=123,
    )
    y = agg.fit_transform(y)

    return y


def test_get_bottom_series_idx(multiindex_df):
    idx = get_bottom_series_idx(multiindex_df)
    assert isinstance(idx, pd.Index)

    assert len(idx) == NUM_BOTTOM_NODES


def test_get_multiindex_loc(multiindex_df):
    df = get_multiindex_loc(multiindex_df, [("l2_node01", "l1_node01")])

    pd.testing.assert_frame_equal(
        df, multiindex_df.loc[pd.IndexSlice["l2_node01", "l1_node01", :],]
    )


def test_loc_bottom_series(multiindex_df):
    df = loc_bottom_series(multiindex_df)

    pd.testing.assert_frame_equal(
        df,
        multiindex_df.loc[
            pd.IndexSlice[("l2_node01",), ("l1_node01", "l1_node02", "l1_node03"), :],
        ],
    )


def test_iterate_all_series(multiindex_df):
    generator = iterate_all_series(multiindex_df)
    all_series = list(generator)

    assert len(all_series) == len(multiindex_df.index.droplevel(-1).unique())

    for idx, series in all_series:
        assert isinstance(idx, tuple)
        assert isinstance(series, pd.DataFrame)


def test_convert_index_to_days_since_epoch():
    idx = pd.date_range(start="1/1/2020", periods=5)
    result = convert_index_to_days_since_epoch(idx)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(idx)


def test_series_to_tensor(multiindex_df):
    result = series_to_tensor(multiindex_df)
    assert isinstance(result, jnp.ndarray)


def test_set_exogenous_priors():
    exogenous_priors = {".*": (dist.Normal, 0, 1)}
    df = pd.DataFrame(np.random.rand(10, 5), columns=list("abcde"))
    result = set_exogenous_priors(exogenous_priors, df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], jnp.ndarray)
    
    with pytest.raises(ValueError):
        set_exogenous_priors({**exogenous_priors, **{"a" : (dist.Normal, 0, 2)}}, df[["a", "b"]])


def test_convert_dataframe_to_tensors(multiindex_df):
    
    result = convert_dataframe_to_tensors(multiindex_df)
    assert isinstance(result, tuple)
    assert isinstance(result[0], jnp.ndarray)
    assert isinstance(result[1], jnp.ndarray)
