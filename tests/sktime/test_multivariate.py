import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import temporal_train_test_split
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen, _make_hierarchical

from prophetverse.effects import LinearEffect
from prophetverse.sktime.multivariate import HierarchicalProphet
from prophetverse.sktime.seasonality import seasonal_transformer

from ._utils import (
    execute_extra_predict_methods_tests,
    execute_fit_predict_test,
    make_empty_X,
    make_None_X,
    make_random_X,
    make_y,
)

HYPERPARAMS = [
    dict(feature_transformer=seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True)),
    dict(
        feature_transformer=seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True),
        default_effect=LinearEffect(effect_mode="multiplicative"),
    ),
    dict(
        feature_transformer=seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True),
        exogenous_effects=[
            LinearEffect(id="lineareffect1", regex=r"(x1).*"),
            LinearEffect(id="lineareffect2", regex=r"(x2).*", prior=dist.Laplace(0, 1)),
        ],
    ),
    dict(
        trend="linear",
    ),
    dict(trend="logistic"),
    dict(inference_method="mcmc"),
    dict(
        feature_transformer=seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True),
        shared_features=["x1"],
    ),
]


@pytest.mark.parametrize("hierarchy_levels", [0, (1,), (2, 1), (1, 2), (3, 2, 2)])
def test_hierarchy_levels(hierarchy_levels):
    y = make_y(hierarchy_levels)
    X = make_random_X(y)
    forecaster = HierarchicalProphet(optimizer_steps=20, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2)
    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_hyperparams(hyperparams):
    hierarchy_levels = (2, 1)
    y = make_y(hierarchy_levels)
    X = make_random_X(y)
    forecaster = HierarchicalProphet(
        **hyperparams, optimizer_steps=20, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2
    )
    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
def test_extra_predict_methods(make_X):
    y = make_y((2, 1))
    X = make_X(y)
    forecaster = HierarchicalProphet(optimizer_steps=20, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2)

    execute_extra_predict_methods_tests(forecaster=forecaster, X=X, y=y)
