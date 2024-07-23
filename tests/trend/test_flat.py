import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from prophetverse.effects.trend.flat import (
    FlatTrend,  # Assuming this is the import path
)


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
    trend_model.fit(X=None, y=timeseries_data)
    expected_loc = timeseries_data["data"].mean()
    assert_almost_equal(trend_model.changepoint_prior_loc, expected_loc)


def test_fit(trend_model, timeseries_data):
    idx = timeseries_data.index
    trend_model.fit(X=None, y=timeseries_data)
    result = trend_model.transform(X=pd.DataFrame(index=timeseries_data.index), fh=idx)
    assert result.shape == (len(idx), 1)
    assert jnp.all(result == 1)


def test_compute_trend(trend_model, timeseries_data):
    idx = timeseries_data.index
    trend_model.fit(X=None, y=timeseries_data)
    constant_vector = trend_model.transform(
        X=pd.DataFrame(index=timeseries_data.index), fh=idx
    )

    with numpyro.handlers.seed(rng_seed=0):
        trend_result = trend_model.predict(constant_vector, None)

    assert jnp.unique(trend_result).shape == (1,)
    assert trend_result.shape == (len(idx), 1)
