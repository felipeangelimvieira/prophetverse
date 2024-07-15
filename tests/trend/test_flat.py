import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from prophetverse.trend.flat import FlatTrend  # Assuming this is the import path


@pytest.fixture
def trend_model():
    return FlatTrend(changepoint_prior_scale=0.1)


@pytest.fixture
def timeseries_data():
    date_rng = pd.date_range(start="1/1/2022", end="1/10/2022", freq="D")
    df = pd.DataFrame(date_rng, columns=["date"])
    df["data"] = jnp.arange(len(date_rng))
    df = df.set_index("date")
    return df


def test_initialization(trend_model):
    assert trend_model.changepoint_prior_scale == 0.1


def test_initialize(trend_model, timeseries_data):
    trend_model.initialize(timeseries_data)
    expected_loc = timeseries_data["data"].mean()
    assert_almost_equal(trend_model.changepoint_prior_loc, expected_loc)


def test_fit(trend_model, timeseries_data):
    idx = timeseries_data.index.to_period("D")
    result = trend_model.fit(idx)
    assert "constant_vector" in result
    assert result["constant_vector"].shape == (len(idx), 1)
    assert jnp.all(result["constant_vector"] == 1)


def test_compute_trend(trend_model, timeseries_data):
    idx = timeseries_data.index.to_period("D")
    trend_model.initialize(timeseries_data)
    data = trend_model.fit(idx)
    constant_vector = data["constant_vector"]

    with numpyro.handlers.seed(rng_seed=0):
        trend_result = trend_model.compute_trend(constant_vector)

    assert jnp.unique(trend_result).shape == (1,)
    assert trend_result.shape == (len(idx), 1)
