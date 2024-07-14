import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest

from prophetverse.effects import LiftExperimentLikelihood, LinearEffect


@pytest.fixture
def lift_test_results():
    return pd.DataFrame(
        data={"test_results": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2021-01-01", periods=6),
    )


@pytest.fixture
def inner_effect():
    return LinearEffect(prior=dist.Delta(2))


@pytest.fixture
def lift_experiment_effect_instance(lift_test_results, inner_effect):
    return LiftExperimentLikelihood(
        effect=inner_effect, lift_test_results=lift_test_results, prior_scale=1.0
    )


@pytest.fixture
def X():
    return pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60]},
        index=pd.date_range("2021-01-01", periods=6),
    )


def test_liftexperimentlikelihood_initialization(
    lift_experiment_effect_instance, inner_effect, lift_test_results
):
    assert lift_experiment_effect_instance.effect == inner_effect
    assert lift_experiment_effect_instance.lift_test_results.equals(lift_test_results)
    assert lift_experiment_effect_instance.prior_scale == 1.0


def test_liftexperimentlikelihood_fit(X, lift_experiment_effect_instance):

    lift_experiment_effect_instance.fit(X, scale=1)
    assert lift_experiment_effect_instance.timeseries_scale == 1
    assert lift_experiment_effect_instance.effect._is_fitted


def test_liftexperimentlikelihood_transform_train(X, lift_experiment_effect_instance):

    lift_experiment_effect_instance.fit(X)
    transformed = lift_experiment_effect_instance.transform(X, stage="train")
    assert "observed_lift" in transformed
    assert transformed["observed_lift"] is not None


def test_liftexperimentlikelihood_transform_predict(X, lift_experiment_effect_instance):
    lift_experiment_effect_instance.fit(X)
    transformed = lift_experiment_effect_instance.transform(X, stage="predict")
    assert "observed_lift" in transformed
    assert transformed["observed_lift"] is None


def test_liftexperimentlikelihood_predict(X, lift_experiment_effect_instance):
    trend = jnp.array([1, 2, 3, 4, 5, 6])

    lift_experiment_effect_instance.fit(X)
    data = lift_experiment_effect_instance.transform(X, stage="train")
    predicted = lift_experiment_effect_instance.predict(trend, **data)
    inner_effect_data = lift_experiment_effect_instance.effect.transform(
        X, stage="train"
    )
    inner_effect_predict = lift_experiment_effect_instance.effect.predict(
        trend, **inner_effect_data
    )
    assert jnp.all(predicted == inner_effect_predict)
