import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import (_bottom_hier_datagen,
                                                _make_hierarchical)

from hierarchical_prophet import exogenous_priors
from hierarchical_prophet.hierarchical_prophet.prophet import HierarchicalProphet

NUM_LEVELS = 2
NUM_BOTTOM_NODES = 3


def _make_random_X(y):
    return pd.DataFrame(np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index)


def _make_None_X(y):
    return None

def _make_empty_X(y):
    return pd.DataFrame(index=y.index)


HYPERPARAMS = [
    dict(exogenous_priors={".*": (dist.Normal, 0, 1)}),
    dict(
        exogenous_priors={
            r"x1": (dist.Normal, 0, 1),
            r"x2": (dist.Laplace, 0, 1),
            r"x3": (dist.Normal, 0, 1),
        },
    ),
    dict(
        exogenous_priors={".*": (dist.Laplace, 0, 1)}
    ),
    dict(
        trend="linear",
    ),
    dict(
        trend="logistic"
    ),
]

@pytest.fixture
def data():
    agg = Aggregator()
    y = _bottom_hier_datagen(
        no_bottom_nodes=NUM_BOTTOM_NODES,
        no_levels=NUM_LEVELS,
        random_seed=123,
    )
    y = agg.fit_transform(y)

    X = pd.DataFrame(
        np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index
    )
    y_train, y_test = (
        y.loc[pd.IndexSlice[:, :, :"1960-01-01"]],
        y.loc[pd.IndexSlice[:, :, "1960-01-01":]],
    )
    return y_train, y_test, X


def test_prophet2_fit_predict(data):
    y_train, y_test, X = data
    fh = list(range(10))

    n_series = len(y_train.index.droplevel(-1).unique())
    forecaster = HierarchicalProphet(
        exogenous_priors={".*": (dist.Normal, 0, 1)}, mcmc_samples=2, mcmc_warmup=2
    )
    forecaster.fit(y_train, X, fh=fh)
    y_pred = forecaster.predict(X=X, fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == 10 * n_series
    assert y_pred.shape[1] == 1


def test_prophet2_fit_predict_with_exogenous_priors(data):
    y_train, y_test, X = data
    fh = list(range(10))
    n_series = len(y_train.index.droplevel(-1).unique())

    exogenous_priors = {
        r"x1": (dist.Normal, 0, 1),
        r"x2": (dist.Laplace, 0, 1),
        r"x3": (dist.Normal, 0, 1),
    }

    forecaster = HierarchicalProphet(
        exogenous_priors=exogenous_priors, mcmc_samples=2, mcmc_warmup=2
    )
    forecaster.fit(y_train, X=X, fh=fh)
    y_pred = forecaster.predict(X=X, fh=fh)
    
    

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * n_series
    assert y_pred.shape[1] == 1


def test_prophet_fit_predict_with_changepoint_prior_scale(data):
    y_train, y_test, X = data
    fh = list(range(10))
    n_series = len(y_train.index.droplevel(-1).unique())

    changepoint_prior_scale = [0.1, 0.2, 0.3]

    forecaster = HierarchicalProphet(
        changepoint_prior_scale=changepoint_prior_scale, mcmc_samples=2, mcmc_warmup=2
    )
    forecaster.fit(y_train, X, fh=fh)
    y_pred = forecaster.predict(X=X, fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * n_series
    assert y_pred.shape[1] == 1



@pytest.mark.parametrize("hierarchy_levels", [(2,), (2, 1), (1, 2), (3, 2, 2)])
@pytest.mark.parametrize("make_X", [_make_random_X, _make_None_X, _make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(hierarchy_levels, make_X, hyperparams):
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels)
    # convert level -1 to pd.periodIndex
    y.index = y.index.set_levels(y.index.levels[-1].to_period("D"), level=-1)

    X = make_X(y)
    fh = list(range(10))

    forecaster = HierarchicalProphet(**hyperparams, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2)
    forecaster.fit(y, X)
    y_pred = forecaster.predict(X=X, fh=fh)

    n_series = len(y.index.droplevel(-1).unique())
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * n_series
    assert y_pred.shape[1] == 1
