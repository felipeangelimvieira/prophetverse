import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects import ExactLikelihood, LinearEffect


@pytest.fixture
def exact_likelihood_results():
    return pd.DataFrame(
        data={"test_results": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2021-01-01", periods=6),
    )


@pytest.fixture
def inner_effect():
    return LinearEffect(prior=dist.Delta(2))


@pytest.fixture
def exact_likelihood_effect_instance(exact_likelihood_results):
    return ExactLikelihood(
        effect_name="exog",
        reference_df=exact_likelihood_results,
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


def test_exact_likelihood_initialization(
    exact_likelihood_effect_instance, exact_likelihood_results
):
    assert exact_likelihood_effect_instance.reference_df.equals(
        exact_likelihood_results
    )
    assert exact_likelihood_effect_instance.prior_scale == 1.0


def test_exact_likelihood_fit(X, exact_likelihood_effect_instance):

    exact_likelihood_effect_instance.fit(y=y, X=X, scale=1)
    assert exact_likelihood_effect_instance.timeseries_scale == 1
    assert exact_likelihood_effect_instance._is_fitted


def test_exact_likelihood_transform_train(X, y, exact_likelihood_effect_instance):
    fh = y.index.get_level_values(-1).unique()
    exact_likelihood_effect_instance.fit(X=X, y=y)
    transformed = exact_likelihood_effect_instance.transform(
        X,
        fh=fh,
    )
    assert "observed_reference_value" in transformed
    assert transformed["observed_reference_value"] is not None


def test_exact_likelihood_predict(X, y, exact_likelihood_effect_instance):
    fh = X.index.get_level_values(-1).unique()

    exog = jnp.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
    exact_likelihood_effect_instance.fit(X=X, y=y)
    data = exact_likelihood_effect_instance.transform(X=X, fh=fh)

    exec_trace = handlers.trace(exact_likelihood_effect_instance.predict).get_trace(
        data=data, predicted_effects={"exog": exog}
    )

    assert len(exec_trace) == 1

    trace_likelihood = exec_trace["exact_likelihood:ignore"]
    assert trace_likelihood["type"] == "sample"
    assert jnp.all(trace_likelihood["value"] == exog)
    assert trace_likelihood["is_observed"]
