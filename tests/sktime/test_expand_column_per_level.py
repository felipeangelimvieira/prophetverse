import numpy as np
import pandas as pd
import pytest

from prophetverse.sktime._expand_column_per_level import ExpandColumnPerLevel


def create_test_dataframe():
    """Helper function to create a test DataFrame with a multi-level index."""
    index = pd.MultiIndex.from_tuples(
        [
            ("series1", "2020-01"),
            ("series1", "2020-02"),
            ("series2", "2020-01"),
            ("series2", "2020-02"),
        ],
        names=["series", "date"],
    )
    return pd.DataFrame(
        {"value1": [1, 2, 3, 4], "value2": [4, 3, 2, 1], "other": [10, 20, 30, 40]},
        index=index,
    )


def test_fit_identifies_matched_columns():
    """
    Test that the fit method correctly identifies columns that match the provided regular expressions.
    """
    X = create_test_dataframe()
    transformer = ExpandColumnPerLevel(columns_regex=["value"])
    transformer.fit(X)

    assert "value1" in transformer.matched_columns_
    assert "value2" in transformer.matched_columns_
    assert "other" not in transformer.matched_columns_

    X = X.loc[("series1")]
    
    transformer = ExpandColumnPerLevel(columns_regex=["value"])
    transformer.fit(X)

    assert "value1" in transformer.matched_columns_
    assert "value2" in transformer.matched_columns_
    assert "other" not in transformer.matched_columns_

def test_transform_expands_columns():
    """
    Test that the transform method correctly expands columns based on the multi-level index.
    """
    X = create_test_dataframe()
    transformer = ExpandColumnPerLevel(columns_regex=["value"])
    transformer.fit(X)
    X_transformed = transformer.transform(X)

    # Check for new columns
    expected_columns = [
        "value1_dup_series1",
        "value1_dup_series2",
        "value2_dup_series1",
        "value2_dup_series2",
        "other",
    ]
    assert all(col in X_transformed.columns for col in expected_columns)


def test_transform_preserves_original_data():
    """
    Test that the transform method preserves the original data in the newly expanded columns.
    """
    X = create_test_dataframe()
    transformer = ExpandColumnPerLevel(columns_regex=["value"])
    transformer.fit(X)
    X_transformed = transformer.transform(X)

    # Check data preservation
    assert X_transformed["value1_dup_series1"].iloc[0] == 1
    assert X_transformed["value2_dup_series1"].iloc[0] == 4
    assert X_transformed["value1_dup_series2"].iloc[2] == 3
    assert X_transformed["value2_dup_series2"].iloc[2] == 2

    # Check for zero filling
    assert X_transformed["value1_dup_series1"].iloc[2] == 0
    assert X_transformed["value2_dup_series1"].iloc[2] == 0
    
    assert (X_transformed.values == 0).sum() == 8
