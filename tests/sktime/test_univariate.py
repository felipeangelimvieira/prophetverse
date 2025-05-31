import pytest
from numpyro import distributions as dist

from prophetverse.effects.linear import LinearEffect
from prophetverse.effects.trend.flat import FlatTrend
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.sktime.univariate import (
    _DISCRETE_LIKELIHOODS,
    _LIKELIHOOD_MODEL_MAP,
    Prophet,
    ProphetGamma,
    ProphetNegBinomial,
    Prophetverse,
)

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

MODELS = [
    Prophet,
    ProphetGamma,
    ProphetNegBinomial,
]

SEASONAL_EFFECT = (
    "seasonality",
    LinearFourierSeasonality(sp_list=[7, 365.25], fourier_terms_list=[1, 1], freq="D"),
    None,
)
HYPERPARAMS = [
    dict(
        trend=FlatTrend(),
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
    dict(trend="logistic"),
    dict(trend="flat"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("model_class", MODELS)
def test_model_class_fit(model_class):
    hierarchy_levels = (1,)
    make_X = make_random_X
    hyperparams = HYPERPARAMS[0]

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(
        **hyperparams,
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.smoke
@pytest.mark.parametrize("hierarchy_levels", [(1,), (2,), (2, 1)])
def test_hierarchy_levels_fit(hierarchy_levels):
    model_class = MODELS[0]
    make_X = make_random_X
    hyperparams = HYPERPARAMS[0]

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(
        **hyperparams,
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.smoke
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_hyperparams_fit(hyperparams):
    model_class = MODELS[1]
    hierarchy_levels = (1,)
    make_X = make_random_X

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(
        **hyperparams,
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
def test_extra_predict_methods(make_X):
    y = make_y((2, 1))
    X = make_X(y)
    forecaster = Prophet(
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )
    execute_extra_predict_methods_tests(forecaster=forecaster, X=X, y=y)


@pytest.mark.ci
@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("hierarchy_levels", [(1,), (2,), (2, 1)])
@pytest.mark.parametrize("make_X", [make_random_X, make_None_X, make_empty_X])
@pytest.mark.parametrize("hyperparams", HYPERPARAMS)
def test_prophet2_fit_with_different_nlevels(
    model_class, hierarchy_levels, make_X, hyperparams
):

    y = make_y(hierarchy_levels)
    X = make_X(y)
    forecaster = model_class(
        **hyperparams,
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        )
    )

    execute_fit_predict_test(forecaster, y, X, test_size=4)


@pytest.mark.parametrize(
    "parameters",
    [
        dict(trend="bad_trend"),
        dict(likelihood="bad_likelihood"),
    ],
)
def test_raise_error_when_passing_parameters(parameters):
    with pytest.raises(ValueError):
        Prophetverse(**parameters)


@pytest.mark.parametrize("likelihood", ["normal", "gamma", "negbinomial"])
def test_prophetverse_likelihood_behaviour(likelihood):
    model = Prophetverse(likelihood=likelihood)
    assert isinstance(model._likelihood, _LIKELIHOOD_MODEL_MAP[likelihood])

    if likelihood in _DISCRETE_LIKELIHOODS:
        assert model._likelihood_is_discrete
