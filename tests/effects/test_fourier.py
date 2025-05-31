import jax.numpy as jnp
import numpyro
import numpy as np
import pandas as pd
import pytest
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects import LinearEffect, LinearFourierSeasonality


@pytest.fixture
def exog_data():
    # Longer period for more comprehensive testing
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=40, freq="D"),
            "value": range(40),
        }
    ).set_index("date")


@pytest.fixture
def y_data(exog_data):
    return pd.DataFrame({"y": np.random.rand(len(exog_data))}, index=exog_data.index)


@pytest.fixture
def fourier_effect_instance():
    return LinearFourierSeasonality(
        sp_list=[365.25],
        fourier_terms_list=[3],
        freq="D",
        prior_scale=1.0,
        effect_mode="additive",
    )


@pytest.fixture
def fourier_effect_instance_config():
    # Config for typical tests, weekly seasonality
    return {
        "sp_list": [7],
        "fourier_terms_list": [3],
        "freq": "D",
        "prior_scale": 1.0,
        "effect_mode": "additive",
    }


def test_linear_fourier_seasonality_initialization(fourier_effect_instance):
    assert fourier_effect_instance.sp_list == [365.25]
    assert fourier_effect_instance.fourier_terms_list == [3]
    assert fourier_effect_instance.freq == "D"
    assert fourier_effect_instance.prior_scale == 1.0
    assert fourier_effect_instance.effect_mode == "additive"
    assert fourier_effect_instance.active_period_start is None
    assert fourier_effect_instance.active_period_end is None


def test_linear_fourier_seasonality_fit(fourier_effect_instance, exog_data):
    fourier_effect_instance.fit(X=exog_data, y=None)
    assert hasattr(fourier_effect_instance, "fourier_features_")
    assert hasattr(fourier_effect_instance, "linear_effect_")
    assert isinstance(fourier_effect_instance.fourier_features_, FourierFeatures)
    assert isinstance(fourier_effect_instance.linear_effect_, LinearEffect)


def test_linear_fourier_seasonality_transform(fourier_effect_instance, exog_data):
    fh = exog_data.index.get_level_values(-1).unique()
    fourier_effect_instance.fit(X=exog_data, y=None)
    transformed = fourier_effect_instance.transform(X=exog_data, fh=fh)

    fourier_transformed = fourier_effect_instance.fourier_features_.transform(exog_data)
    assert isinstance(transformed, jnp.ndarray)
    assert transformed.shape == fourier_transformed.shape


def test_linear_fourier_seasonality_predict(fourier_effect_instance, exog_data):
    fh = exog_data.index.get_level_values(-1).unique()
    fourier_effect_instance.fit(X=exog_data, y=None)
    trend = jnp.array([1.0] * len(exog_data))
    data = fourier_effect_instance.transform(exog_data, fh=fh)
    with numpyro.handlers.seed(numpyro.handlers.seed, 0):
        prediction = fourier_effect_instance.predict(
            data, predicted_effects={"trend": trend}
        )
    assert prediction is not None
    assert isinstance(prediction, jnp.ndarray)


def test_active_period_middle(fourier_effect_instance_config, exog_data, y_data):
    start_date = pd.to_datetime("2021-01-10")
    end_date = pd.to_datetime("2021-01-20")
    
    effect = LinearFourierSeasonality(
        **fourier_effect_instance_config,
        active_period_start=start_date,
        active_period_end=end_date
    )
    effect.fit(X=exog_data, y=y_data)
    transformed_effect = effect.transform(X=exog_data, fh=exog_data.index)

    assert isinstance(transformed_effect, jnp.ndarray)
    
    before_mask = exog_data.index < start_date
    during_mask = (exog_data.index >= start_date) & (exog_data.index <= end_date)
    after_mask = exog_data.index > end_date

    assert jnp.all(transformed_effect[before_mask] == 0), "Effect should be zero before active period"
    # Sum of absolute values to check for non-zero, as individual terms can be zero
    assert jnp.sum(jnp.abs(transformed_effect[during_mask])) > 1e-6, "Effect should be non-zero during active period"
    assert jnp.all(transformed_effect[after_mask] == 0), "Effect should be zero after active period"


def test_active_period_start_only(fourier_effect_instance_config, exog_data, y_data):
    start_date = pd.to_datetime("2021-01-15")
    effect = LinearFourierSeasonality(
        **fourier_effect_instance_config,
        active_period_start=start_date
    )
    effect.fit(X=exog_data, y=y_data)
    transformed_effect = effect.transform(X=exog_data, fh=exog_data.index)

    before_mask = exog_data.index < start_date
    during_mask = exog_data.index >= start_date
    
    assert jnp.all(transformed_effect[before_mask] == 0)
    assert jnp.sum(jnp.abs(transformed_effect[during_mask])) > 1e-6

def test_active_period_end_only(fourier_effect_instance_config, exog_data, y_data):
    end_date = pd.to_datetime("2021-01-15")
    effect = LinearFourierSeasonality(
        **fourier_effect_instance_config,
        active_period_end=end_date
    )
    effect.fit(X=exog_data, y=y_data)
    transformed_effect = effect.transform(X=exog_data, fh=exog_data.index)

    during_mask = exog_data.index <= end_date
    after_mask = exog_data.index > end_date

    assert jnp.sum(jnp.abs(transformed_effect[during_mask])) > 1e-6
    assert jnp.all(transformed_effect[after_mask] == 0)
