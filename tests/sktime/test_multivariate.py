import pytest
from numpyro import distributions as dist

from prophetverse.effects.linear import LinearEffect
from prophetverse.effects.trend import PiecewiseLinearTrend
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
    dict(
        trend=PiecewiseLinearTrend(
            changepoint_interval=20,
            changepoint_range=0.8,
            changepoint_prior_scale=0.001,
        ),
        feature_transformer=seasonal_transformer(
            yearly_seasonality=True, weekly_seasonality=True
        ),
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
            ("lineareffect1", LinearEffect(), r"(x1).*"),
            ("lineareffect1_repeated", LinearEffect(), r"(x1).*"),
            ("lineareffect2", LinearEffect(prior=dist.Laplace(0, 1)), r"(x2).*"),
            (
                "lineareffect_no_match",
                LinearEffect(prior=dist.Laplace(0, 1)),
                r"(x10).*",
            ),
        ],
    ),
    dict(
        trend="linear_raw",
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


@pytest.mark.smoke
@pytest.mark.parametrize("hierarchy_levels", [0, (1,), (2, 1), (1, 2), (3, 2, 2)])
def test_hierarchy_levels(hierarchy_levels):
    y = make_y(hierarchy_levels)
    X = make_random_X(y)
    forecaster = HierarchicalProphet(
        optimizer_steps=2, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2
    )
    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.smoke
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_hyperparams(hyperparams):
    hierarchy_levels = (2, 1)
    y = make_y(hierarchy_levels)
    X = make_random_X(y)
    forecaster = HierarchicalProphet(
        **hyperparams,
        optimizer_steps=2,
        changepoint_interval=2,
        mcmc_samples=2,
        mcmc_warmup=2
    )
    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.ci
@pytest.mark.parametrize("hierarchy_levels", [0, (1,), (2, 1), (1, 2), (3, 2, 2)])
@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(hierarchy_levels, make_X, hyperparams):
    y = make_y(hierarchy_levels)

    X = make_X(y)

    forecaster = HierarchicalProphet(
        **hyperparams,
        optimizer_steps=20,
        changepoint_interval=2,
        mcmc_samples=2,
        mcmc_warmup=2
    )

    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
def test_extra_predict_methods(make_X):
    y = make_y((2, 1))
    X = make_X(y)
    forecaster = HierarchicalProphet(
        optimizer_steps=20, changepoint_interval=2, mcmc_samples=2, mcmc_warmup=2
    )

    execute_extra_predict_methods_tests(forecaster=forecaster, X=X, y=y)
