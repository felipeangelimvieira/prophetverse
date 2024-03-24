import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import (_bottom_hier_datagen,
                                                _make_hierarchical)

from hierarchical_prophet import exogenous_priors
from hierarchical_prophet.prophet import Prophet

NUM_LEVELS = 2
NUM_BOTTOM_NODES = 3


def _make_random_X(y):
    return pd.DataFrame(np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index)


def _make_None_X(y):
    return None

def _make_empty_X(y):
    return pd.DataFrame(index=y.index)


HYPERPARAMS = [
    dict(yearly_seasonality=True, weekly_seasonality=True),
    dict(yearly_seasonality=False, weekly_seasonality=True),
    dict(yearly_seasonality=5, weekly_seasonality=2),
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


@pytest.mark.parametrize("hierarchy_levels", [(1,), (2,), (2, 1)])
@pytest.mark.parametrize("make_X", [_make_random_X, _make_None_X, _make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(hierarchy_levels, make_X, hyperparams):
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels)
    # convert level -1 to pd.periodIndex
    y.index = y.index.set_levels(y.index.levels[-1].to_period("D"), level=-1)

    X = make_X(
        y
    )
    fh = list(range(-5, 3))
    y_train = y.loc[y.index.get_level_values(-1) < "2000-01-10"]
    forecaster = Prophet(**hyperparams, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)
    forecaster.fit(y_train, X)
    y_pred = forecaster.predict(X=X, fh=fh)

    n_series = len(y.index.droplevel(-1).unique())
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * n_series
    assert y_pred.shape[1] == 1
