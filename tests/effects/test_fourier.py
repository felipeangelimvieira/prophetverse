import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects import LinearEffect, LinearFourierSeasonality


@pytest.fixture
def exog_data():
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=10),
            "value": range(10),
        }
    ).set_index("date")


@pytest.fixture
def fourier_effect_instance():
    return LinearFourierSeasonality(
        sp_list=[365.25],
        fourier_terms_list=[3],
        freq="D",
        prior_scale=1.0,
        effect_mode="additive",
    )


def test_linear_fourier_seasonality_initialization(fourier_effect_instance):
    assert fourier_effect_instance.sp_list == [365.25]
    assert fourier_effect_instance.fourier_terms_list == [3]
    assert fourier_effect_instance.freq == "D"
    assert fourier_effect_instance.prior_scale == 1.0
    assert fourier_effect_instance.effect_mode == "additive"


def test_linear_fourier_seasonality_fit(fourier_effect_instance, exog_data):
    fourier_effect_instance.fit(exog_data)
    assert hasattr(fourier_effect_instance, "fourier_features_")
    assert hasattr(fourier_effect_instance, "linear_effect_")
    assert isinstance(fourier_effect_instance.fourier_features_, FourierFeatures)
    assert isinstance(fourier_effect_instance.linear_effect_, LinearEffect)


def test_linear_fourier_seasonality_transform(fourier_effect_instance, exog_data):
    fourier_effect_instance.fit(exog_data)
    transformed = fourier_effect_instance.transform(exog_data, stage="train")

    fourier_transformed = fourier_effect_instance.fourier_features_.transform(exog_data)
    assert isinstance(transformed, dict)
    assert "data" in transformed
    assert transformed["data"].shape == fourier_transformed.shape


def test_linear_fourier_seasonality_predict(fourier_effect_instance, exog_data):
    fourier_effect_instance.fit(exog_data)
    trend = jnp.array([1.0] * len(exog_data))
    data = fourier_effect_instance.transform(exog_data, stage="predict")
    with numpyro.handlers.seed(numpyro.handlers.seed, 0):
        prediction = fourier_effect_instance.predict(trend, **data)
    assert prediction is not None
    assert isinstance(prediction, jnp.ndarray)
