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


@pytest.fixture
def y(X):
    return pd.DataFrame(index=X.index, data=[1] * len(X))


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


def test_liftexperimentlikelihood_transform_train(
    X, y, lift_experiment_effect_instance
):
    fh = y.index.get_level_values(-1).unique()
    lift_experiment_effect_instance.fit(X, y=y)
    transformed = lift_experiment_effect_instance.transform(
        X,
        fh=fh,
    )
    assert "observed_lift" in transformed
    assert transformed["observed_lift"] is not None


def test_liftexperimentlikelihood_predict(X, y, lift_experiment_effect_instance):
    fh = X.index.get_level_values(-1).unique()

    trend = jnp.array([1, 2, 3, 4, 5, 6])
    lift_experiment_effect_instance.fit(X)
    data = lift_experiment_effect_instance.transform(X, fh=fh)
    predicted = lift_experiment_effect_instance.predict(
        data=data, predicted_effects={"trend": trend}
    )
    inner_effect_data = lift_experiment_effect_instance.effect.transform(X, fh=fh)
    inner_effect_predict = lift_experiment_effect_instance.effect.predict(
        data=inner_effect_data, predicted_effects={"trend": trend}
    )
    assert jnp.all(predicted == inner_effect_predict)
