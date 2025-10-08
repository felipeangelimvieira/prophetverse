import pandas as pd
import pytest

from prophetverse.effects.measurements import generate_scenarious


def _base_dataframe():
    index = pd.date_range("2020-01-01", periods=3)
    return pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]}, index=index)


def test_generate_scenarious_fills_missing_values():
    X = _base_dataframe()
    counterfactual = pd.DataFrame(
        {"A": [9, 7, 3], "B": [2, None, None], "C": [9, 1, 0]},
        index=X.index,
    )

    result = generate_scenarious(X, [counterfactual])

    expected = pd.DataFrame(
        {"A": [9, 7, 3], "B": [2, 5, 8], "C": [9, 1, 0]},
        index=X.index,
    )

    assert len(result) == 1
    assert result[0].shape == X.shape
    pd.testing.assert_frame_equal(result[0], expected)


def test_generate_scenarious_partial_index_and_columns():
    X = _base_dataframe()
    counterfactual = pd.DataFrame({"A": [100, 200]}, index=X.index[:2])

    result = generate_scenarious(X, [counterfactual])

    expected = X.copy()
    expected.loc[X.index[0], "A"] = 100
    expected.loc[X.index[1], "A"] = 200

    assert result[0].shape == X.shape
    pd.testing.assert_frame_equal(result[0], expected)


def test_generate_scenarious_raises_for_unknown_columns():
    X = _base_dataframe()
    counterfactual = pd.DataFrame({"Z": [1, 2, 3]}, index=X.index)

    with pytest.raises(ValueError):
        generate_scenarious(X, [counterfactual])


def test_generate_scenarious_requires_dataframe():
    X = _base_dataframe()

    with pytest.raises(TypeError):
        generate_scenarious(X, [{"A": [1, 2, 3]}])
