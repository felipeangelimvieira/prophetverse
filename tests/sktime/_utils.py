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

EXTRA_FORECAST_FUNCS = [
    "predict_interval",
    "predict_all_sites",
    "predict_all_sites_samples",
    "predict_samples",
]

def execute_fit_predict_test(forecaster, y, X, test_size=4):

    y_train, y_test, X_train, X_test = _split_train_test(y, X, test_size=test_size)

    fh = list(range(1, test_size + 1))
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


def _split_train_test(y, X, test_size=4):
    
    dataset = temporal_train_test_split(y, X, test_size=test_size)
    if X is not None:
        y_train, y_test, X_train, X_test = dataset
    else:
        y_train, y_test = dataset
        X_train, X_test = None, None

    return y_train, y_test, X_train, X_test

def make_random_X(y):
    return pd.DataFrame(
        np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index
    )


def make_None_X(y):
    return None


def make_empty_X(y):
    return pd.DataFrame(index=y.index)


def make_y(hierarchy_levels):
    if hierarchy_levels == 0:
        y = _make_hierarchical(
            hierarchy_levels=(1,), max_timepoints=12, min_timepoints=12
        ).droplevel(0) 
    else:
        y = _make_hierarchical(
            hierarchy_levels=hierarchy_levels, max_timepoints=12, min_timepoints=12
        )
        y = Aggregator().fit_transform(y)
        # convert level -1 to pd.periodIndex
        y.index = y.index.set_levels(y.index.levels[-1].to_period("D"), level=-1)
    return y


def execute_extra_predict_methods_tests(forecaster, y, X, test_size=4):

    y_train, y_test, X_train, X_test = _split_train_test(y, X, test_size=test_size)

    fh = y_test.index.get_level_values(-1).unique()
    forecaster.fit(y_train, X_train)

    n_series = y_train.index.droplevel(-1).nunique()
    for forecast_func in EXTRA_FORECAST_FUNCS:
        preds =  getattr(forecaster, forecast_func)(X=X, fh=fh) 
        assert preds is not None
        assert isinstance(preds, pd.DataFrame)

        # TODO: Add more checks for the shape of the predictions