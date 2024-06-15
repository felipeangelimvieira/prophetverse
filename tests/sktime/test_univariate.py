import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen, _make_hierarchical

from prophetverse.effects import LinearEffect
from prophetverse.sktime.seasonality import seasonal_transformer
from prophetverse.sktime.univariate import Prophet, ProphetGamma, ProphetNegBinomial
from prophetverse.trend.flat import FlatTrend

from ._utils import (
    execute_extra_predict_methods_tests,
    execute_fit_predict_test,
    make_empty_X,
    make_None_X,
    make_random_X,
    make_y,
)

MODELS = [
    Prophet,
    ProphetGamma,
    ProphetNegBinomial,
]

HYPERPARAMS = [
    dict(trend=FlatTrend(), feature_transformer=seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True)),
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
    dict(trend="logistic", offset_prior_scale=0.5),
    dict(trend="flat"),
    dict(inference_method="mcmc"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("model_class", MODELS)
def test_model_class_fit(model_class):
    hierarchy_levels = (1,)
    make_X = make_random_X
    hyperparams = HYPERPARAMS[0]

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(**hyperparams, optimizer_steps=10, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.smoke
@pytest.mark.parametrize("hierarchy_levels", [(1,), (2,), (2, 1)])
def test_hierarchy_levels_fit(hierarchy_levels):
    model_class = MODELS[0]
    make_X = make_random_X
    hyperparams = HYPERPARAMS[0]

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(**hyperparams, optimizer_steps=10, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.smoke
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_hyperparams_fit(hyperparams):
    model_class = MODELS[1]
    hierarchy_levels = (1,)
    make_X = make_random_X

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(**hyperparams, optimizer_steps=10, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
def test_extra_predict_methods(make_X):
    y = make_y((2, 1))
    X = make_X(y)
    forecaster = Prophet(optimizer_steps=10, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)
    execute_extra_predict_methods_tests(forecaster=forecaster, X=X, y=y)


@pytest.mark.ci
@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("hierarchy_levels", [(1,), (2,), (2, 1)])
@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(model_class, hierarchy_levels, make_X, hyperparams):

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(**hyperparams, optimizer_steps=100, mcmc_samples=2, mcmc_warmup=2, mcmc_chains=1)

    execute_fit_predict_test(forecaster, y, X, test_size=4)


def test_raise_error_when_passing_bad_trend():
    with pytest.raises(ValueError):
        Prophet(trend="bad_trend")
