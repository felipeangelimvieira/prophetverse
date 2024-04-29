import pytest
import pandas as pd
from prophetverse.sktime.seasonality import (
    seasonal_transformer,
) 

from sktime.transformations.series.fourier import FourierFeatures


def test_valid_input_boolean():
    # Test for valid boolean inputs
    transformer = seasonal_transformer(yearly_seasonality=True, weekly_seasonality=True)
    assert isinstance(
        transformer, FourierFeatures
    ), "Output should be FourierFeatures"


def test_valid_input_integer():
    # Test for valid integer inputs
    transformer = seasonal_transformer(yearly_seasonality=12, weekly_seasonality=4)
    assert isinstance(
        transformer, FourierFeatures
    ), "Output should be FourierFeatures"


def test_invalid_input():
    # Test for invalid inputs
    with pytest.raises(ValueError):
        _ = seasonal_transformer(yearly_seasonality="yearly", weekly_seasonality=3)
    with pytest.raises(ValueError):
        _ = seasonal_transformer(yearly_seasonality=10, weekly_seasonality="weekly")


def test_no_seasonality():
    # Test with no seasonality
    transformer = seasonal_transformer(
        yearly_seasonality=False, weekly_seasonality=False
    )
    assert isinstance(
        transformer, FourierFeatures
    ), "Output should be FourierFeatures"


def test_mixed_validity():
    # Test with one valid and one invalid input
    with pytest.raises(ValueError):
        _ = seasonal_transformer(yearly_seasonality=True, weekly_seasonality="some")
    with pytest.raises(ValueError):
        _ = seasonal_transformer(yearly_seasonality="some", weekly_seasonality=True)


def test_frequency_handling():
    # Test different frequency inputs
    transformer = seasonal_transformer(freq="M")
    assert (
        transformer.get_params()["freq"] == "M"
    ), "Frequency parameter not set correctly"
