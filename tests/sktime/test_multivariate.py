import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import temporal_train_test_split
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import (_bottom_hier_datagen,
                                                _make_hierarchical)

from prophetverse.effects import LinearEffect
from prophetverse.sktime.multivariate import HierarchicalProphet
from prophetverse.sktime.seasonality import seasonal_transformer

NUM_LEVELS = 2
NUM_BOTTOM_NODES = 3

EXTRA_FORECAST_FUNCS = [
    "predict_interval",
    "predict_all_sites",
    "predict_all_sites_samples",
    "predict_samples",
]


def _make_random_X(y):
    return pd.DataFrame(
        np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index
    )


def _make_None_X(y):
    return None


def _make_empty_X(y):
    return pd.DataFrame(index=y.index)


HYPERPARAMS = [
    dict(
        feature_transformer=seasonal_transformer(
            yearly_seasonality=True, weekly_seasonality=True
        )
    ),
    dict(
        feature_transformer=seasonal_transformer(
            yearly_seasonality=True, weekly_seasonality=True
        ),
        default_effect=LinearEffect(effect_mode="multiplicative"),
    ),
    dict(
        feature_transformer=seasonal_transformer(
            yearly_seasonality=True, weekly_seasonality=True
        ),
        exogenous_effects=[
            LinearEffect(id="lineareffect1", regex=r"(x1).*"),
            LinearEffect(
                id="lineareffect2", regex=r"(x2).*", prior=dist.Laplace(0, 1)
            ),
        ],
    ),
    dict(
        trend="linear",
    ),
    dict(trend="logistic"),
    dict(inference_method="mcmc"),
    dict(
        feature_transformer=seasonal_transformer(
            yearly_seasonality=True, weekly_seasonality=True
        ),
        shared_features=["x1"],
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


def _execute_test(forecaster, y, X, test_size=4):

    dataset  =temporal_train_test_split(y, X,test_size=test_size)
    if X is not None:
        y_train, y_test, X_train, X_test = dataset
    else:
        y_train, y_test = dataset
        X_train, X_test = None, None

    fh = list(range(1, test_size+1))
    forecaster.fit(y_train, X_train)
    y_pred = forecaster.predict(X=X_test, fh=fh)

    if y.index.nlevels == 1:
        n_series = 1
    else:
        n_series = len(y.index.droplevel(-1).unique())
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * n_series
    assert y_pred.shape[1] == 1
    assert all(y_pred.index == y_test.index)

    for forecast_func in EXTRA_FORECAST_FUNCS:
        assert getattr(forecaster, forecast_func)(X=X, fh=fh) is not None

@pytest.mark.parametrize("hierarchy_levels", [0, (1,), (2, 1), (1, 2), (3, 2, 2)])
@pytest.mark.parametrize("make_X", [_make_random_X, _make_None_X, _make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(hierarchy_levels, make_X, hyperparams):
    if hierarchy_levels == 0:
        y = _make_hierarchical(hierarchy_levels=(1,), max_timepoints=12, min_timepoints=12).droplevel(0)
    else:
        y = _make_hierarchical(hierarchy_levels=hierarchy_levels, max_timepoints=12, min_timepoints=12)
        y = Aggregator().fit_transform(y)
        # convert level -1 to pd.periodIndex
        y.index = y.index.set_levels(y.index.levels[-1].to_period("D"), level=-1)

    X = make_X(y)

    forecaster = HierarchicalProphet(
        **hyperparams,
        optimizer_steps=20,
        changepoint_interval=2,
        mcmc_samples=2,
        mcmc_warmup=2
    )
    
    _execute_test(forecaster, y, X)
