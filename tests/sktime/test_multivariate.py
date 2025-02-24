import pytest
from numpyro import distributions as dist

from prophetverse.effects.linear import LinearEffect

from prophetverse.effects import LinearFourierSeasonality
from prophetverse.sktime.multivariate import HierarchicalProphet
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import AdamOptimizer

from ._utils import (
    execute_extra_predict_methods_tests,
    execute_fit_predict_test,
    make_empty_X,
    make_None_X,
    make_random_X,
    make_y,
)

SEASONAL_EFFECT = (
    "seasonality",
    LinearFourierSeasonality(sp_list=[7, 365.25], fourier_terms_list=[1, 1], freq="D"),
    None,
)

HYPERPARAMS = [
    dict(
        exogenous_effects=[SEASONAL_EFFECT],
    ),
    dict(
        exogenous_effects=[SEASONAL_EFFECT],
        default_effect=LinearEffect(effect_mode="multiplicative"),
    ),
    dict(
        exogenous_effects=[
            SEASONAL_EFFECT,
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
        trend="linear",
    ),
    dict(
        trend="logistic",
    ),
    dict(),
    dict(
        exogenous_effects=[
            SEASONAL_EFFECT,
        ],
        shared_features=["x1"],
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize("hierarchy_levels", [0, (1,), (2, 1), (1, 2), (3, 2, 2)])
def test_hierarchy_levels(hierarchy_levels):
    y = make_y(hierarchy_levels)
    X = make_random_X(y)
    forecaster = HierarchicalProphet(
        trend="linear",
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
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
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
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
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
    )

    execute_fit_predict_test(forecaster, y, X)


@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
def test_extra_predict_methods(make_X):
    y = make_y((2, 1))
    X = make_X(y)
    forecaster = HierarchicalProphet(
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
    )

    execute_extra_predict_methods_tests(forecaster=forecaster, X=X, y=y)


def test_hierarchical_with_series_with_zeros():
    y = make_y((2, 2, 1))
    # Set all values to 0
    y.iloc[:, :] = 0

    forecaster = HierarchicalProphet(
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )

    forecaster.fit(y)
    forecaster.predict(fh=[1, 2, 3])
