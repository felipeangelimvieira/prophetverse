import jax.numpy as jnp
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects import ROILikelihood


@pytest.fixture
def X():
    return pd.DataFrame(
        data={"spend": [10, 20, 30, 40, 50, 60]},
        index=pd.date_range("2021-01-01", periods=6),
    )


@pytest.fixture
def y(X):
    return pd.DataFrame(index=X.index, data=[1] * len(X))


@pytest.fixture
def roi_likelihood_instance():
    return ROILikelihood(
        effect_name="marketing_effect",
        roi_mean=2.0,
        roi_scale=0.5,
    )


def test_roi_likelihood_initialization(roi_likelihood_instance):
    assert roi_likelihood_instance.effect_name == "marketing_effect"
    assert roi_likelihood_instance.roi_mean == 2.0
    assert roi_likelihood_instance.roi_scale == 0.5


def test_roi_likelihood_scale_must_be_positive():
    with pytest.raises(AssertionError, match="roi_scale must be greater than 0"):
        ROILikelihood(effect_name="test", roi_mean=1.0, roi_scale=-0.5)


def test_roi_likelihood_fit(X, y, roi_likelihood_instance):
    roi_likelihood_instance.fit(y=y, X=X, scale=1.5)
    assert roi_likelihood_instance.timeseries_scale == 1.5
    assert roi_likelihood_instance._is_fitted


def test_roi_likelihood_transform(X, y, roi_likelihood_instance):
    fh = y.index.get_level_values(-1).unique()
    roi_likelihood_instance.fit(X=X, y=y)
    transformed = roi_likelihood_instance.transform(X, fh=fh)

    assert "input_values" in transformed
    assert "input_sum" in transformed
    # Sum of [10, 20, 30, 40, 50, 60] = 210
    assert jnp.isclose(transformed["input_sum"], 210.0)


def test_roi_likelihood_predict(X, y, roi_likelihood_instance):
    fh = X.index.get_level_values(-1).unique()

    # Simulate effect contribution that returns 2x the input (ROI = 2.0)
    # Input sum = 210, so effect sum should be 420 for ROI = 2.0
    effect_contribution = jnp.array([20, 40, 60, 80, 100, 120]).reshape((-1, 1))

    roi_likelihood_instance.fit(X=X, y=y)
    data = roi_likelihood_instance.transform(X=X, fh=fh)

    exec_trace = handlers.trace(roi_likelihood_instance.predict).get_trace(
        data=data, predicted_effects={"marketing_effect": effect_contribution}
    )

    # Check that the likelihood sample was created
    assert "roi_likelihood:ignore" in exec_trace

    trace_likelihood = exec_trace["roi_likelihood:ignore"]
    assert trace_likelihood["type"] == "sample"
    assert trace_likelihood["is_observed"]

    # Check that the observed ROI is correct (420 / 210 = 2.0)
    observed_roi = trace_likelihood["value"]
    assert jnp.isclose(observed_roi, 2.0)


def test_roi_likelihood_returns_zeros(X, y, roi_likelihood_instance):
    fh = X.index.get_level_values(-1).unique()

    effect_contribution = jnp.array([20, 40, 60, 80, 100, 120]).reshape((-1, 1))

    roi_likelihood_instance.fit(X=X, y=y)
    data = roi_likelihood_instance.transform(X=X, fh=fh)

    result = roi_likelihood_instance.predict(
        data=data, predicted_effects={"marketing_effect": effect_contribution}
    )

    # Should return zeros with same shape as effect_contribution
    assert result.shape == effect_contribution.shape
    assert jnp.all(result == 0)
