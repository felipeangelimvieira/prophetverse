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
import pandas as pd
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.effects.trend import PiecewiseLogisticTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.utils import no_input_columns


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


@pytest.mark.parametrize("likelihood", ["normal", "gamma", "negbinomial", "beta"])
def test_prophetverse_likelihood_behaviour(likelihood):
    model = Prophetverse(likelihood=likelihood)
    assert isinstance(model._likelihood, _LIKELIHOOD_MODEL_MAP[likelihood])

    if likelihood in _DISCRETE_LIKELIHOODS:
        assert model._likelihood_is_discrete


def test_prophetverse_hierarchical_with_series_with_zeros():
    y = make_y((2, 2, 5))
    # Set all values to 0
    y.iloc[:, :] = 0

    forecaster = Prophetverse(
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
        broadcast_mode="effect",
    )

    forecaster.fit(y)
    forecaster.predict(fh=[1, 2, 3])


def test_broadcast_does_not_raise_error():
    from prophetverse.datasets.loaders import load_tourism

    y = load_tourism(groupby="Purpose")

    model = Prophetverse(
        trend=PiecewiseLogisticTrend(
            changepoint_prior_scale=0.1,
            changepoint_interval=8,
            changepoint_range=-8,
        ),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(
                    sp_list=["Y"],
                    fourier_terms_list=[1],
                    freq="Q",
                    prior_scale=0.1,
                    effect_mode="multiplicative",
                ),
                no_input_columns,
            )
        ],
        inference_engine=MAPInferenceEngine(),
        broadcast_mode="effect",
    )
    model.fit(y=y)

    forecast_horizon = pd.period_range("1997Q1", "2020Q4", freq="Q")
    preds = model.predict(fh=forecast_horizon)

    # assert all values are finite
    assert preds.isna().sum().sum() == 0


@pytest.mark.smoke
def test_target_likelihood_as_exogenous_effect():
    """Test that target likelihoods can be used as exogenous effects.

    This test verifies the fix for the bug where _transform_effects was not
    receiving the y parameter in _get_fit_data, which prevented target
    likelihoods from being used as exogenous effects.
    """
    from prophetverse.effects.target.univariate import NormalTargetLikelihood

    y = make_y((1,))

    # Create a model with a target likelihood as an exogenous effect
    model = Prophetverse(
        trend="linear",
        exogenous_effects=[
            (
                "target_likelihood",
                NormalTargetLikelihood(noise_scale=0.1),
                None,  # No regex pattern, applies to y
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
    )

    # This should not raise an error - the bug would cause AttributeError
    # when trying to call y.copy() on None
    model.fit(y=y)
    preds = model.predict(fh=[1, 2, 3])

    # Verify predictions are valid
    assert preds is not None
    assert not preds.isna().any().any()
