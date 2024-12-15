import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects import LiftExperimentLikelihood, LinearEffect


@pytest.fixture
def lift_test_results():
    index = pd.date_range("2021-01-01", periods=2)
    return pd.DataFrame(
        index=index, data={"x_start": [1, 2], "x_end": [10, 20], "lift": [2, 4]}
    )


@pytest.fixture
def inner_effect():
    return LinearEffect(prior=dist.Delta(2), effect_mode="additive")


@pytest.fixture
def lift_experiment_likelihood_effect_instance(lift_test_results, inner_effect):
    return LiftExperimentLikelihood(
        effect=inner_effect,
        lift_test_results=lift_test_results,
        prior_scale=1.0,
    )


@pytest.fixture
def X():
    return pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60]},
        index=pd.date_range("2021-01-01", periods=6),
    )


@pytest.fixture
def y(X):
    return pd.DataFrame(index=X.index, data=[1] * len(X))


def test_lift_experiment_likelihood_initialization(
    lift_experiment_likelihood_effect_instance, lift_test_results
):
    assert lift_experiment_likelihood_effect_instance.lift_test_results.equals(
        lift_test_results
    )
    assert lift_experiment_likelihood_effect_instance.prior_scale == 1.0


def test_lift_experiment_likelihood_fit(X, lift_experiment_likelihood_effect_instance):

    lift_experiment_likelihood_effect_instance.fit(y=y, X=X, scale=1)
    assert lift_experiment_likelihood_effect_instance.timeseries_scale == 1
    assert lift_experiment_likelihood_effect_instance.effect_._is_fitted


def test_lift_experiment_likelihood_transform_train(
    X, y, lift_experiment_likelihood_effect_instance, lift_test_results
):
    fh = y.index.get_level_values(-1).unique()
    lift_experiment_likelihood_effect_instance.fit(X=X, y=y)
    transformed = lift_experiment_likelihood_effect_instance.transform(
        X,
        fh=fh,
    )
    assert "observed_lift" in transformed
    assert len(transformed["observed_lift"]) == len(lift_test_results)


def test_lift_experiment_likelihood_predict(
    X, y, lift_experiment_likelihood_effect_instance
):
    fh = X.index.get_level_values(-1).unique()

    exog = jnp.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    lift_experiment_likelihood_effect_instance.fit(X=X, y=y)
    data = lift_experiment_likelihood_effect_instance.transform(X=X, fh=fh)

    exec_trace = handlers.trace(
        lift_experiment_likelihood_effect_instance.predict
    ).get_trace(data=data, predicted_effects={"exog": exog})

    assert "lift_experiment:ignore" in exec_trace
    trace_likelihood = exec_trace["lift_experiment:ignore"]
    assert trace_likelihood["type"] == "sample"
    assert trace_likelihood["is_observed"]
