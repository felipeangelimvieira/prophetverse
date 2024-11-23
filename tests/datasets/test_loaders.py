import pandas as pd

# Import the functions to test
from prophetverse.datasets.loaders import (
    load_peyton_manning,
    load_tensorflow_github_stars,
    load_tourism,
)


def test_load_tourism():
    # Test default loading
    y = load_tourism()
    assert isinstance(y, pd.DataFrame), "load_tourism should return a DataFrame"
    assert not y.empty, "DataFrame should not be empty"
    assert isinstance(y.index, pd.MultiIndex), "Index should be a MultiIndex"
    expected_levels = ["Region", "Quarter"]
    assert y.index.names == expected_levels, f"Index levels should be {expected_levels}"

    # Test with different groupby parameters
    y_region = load_tourism(groupby="Region")
    assert y_region.index.names == [
        "Region",
        "Quarter",
    ], "Index levels should match groupby parameter"

    y_state_purpose = load_tourism(groupby=["State", "Purpose"])
    assert y_state_purpose.index.names == [
        "State",
        "Purpose",
        "Quarter",
    ], "Index levels should match groupby parameters"


def test_load_peyton_manning():
    df = load_peyton_manning()
    assert isinstance(df, pd.DataFrame), "load_peyton_manning should return a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert "y" in df.columns, "DataFrame should contain 'y' column"
    assert isinstance(df.index, pd.PeriodIndex), "Index should be a PeriodIndex"
    assert df.index.freq == "D", "Index frequency should be daily"


def test_load_tensorflow_github_stars():
    y = load_tensorflow_github_stars()
    assert isinstance(
        y, pd.DataFrame
    ), "load_tensorflow_github_stars should return a DataFrame"
    assert not y.empty, "DataFrame should not be empty"
    assert "day-stars" in y.columns, "DataFrame should contain 'day-stars' column"
    assert isinstance(y.index, pd.PeriodIndex), "Index should be a PeriodIndex"
    assert y.index.freq == "D", "Index frequency should be daily"
