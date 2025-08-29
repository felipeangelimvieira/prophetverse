import pandas as pd
import numpy as np
import pytest

from prophetverse.utils.pandas import _build_full_index


@pytest.mark.parametrize(
    "start,end,freq",
    [
        ("2024-01-01", "2024-01-05", "D"),
        ("2024-01-01", "2024-01-01", "D"),
    ],
)
def test_build_full_index_datetime(start, end, freq):
    idx = pd.date_range(start=start, periods=3, freq=freq)
    full = _build_full_index(index=idx, index_start=start, index_end=end)
    assert full[0] == pd.Timestamp(start)
    assert full[-1] == pd.Timestamp(end)
    if start != end:
        # length matches inclusive range with freq
        expected_len = (
            pd.Timestamp(end) - pd.Timestamp(start)
        ) / pd.tseries.frequencies.to_offset(freq) + 1
        assert len(full) == int(expected_len)


def test_build_full_index_period():
    idx = pd.period_range(start="2024-01", periods=3, freq="M")
    full = _build_full_index(index=idx, index_start=idx[0], index_end=idx[-1])
    assert isinstance(full, pd.PeriodIndex)
    assert full[0] == idx[0]
    assert full[-1] == idx[-1]


def test_build_full_index_timedelta():
    idx = pd.timedelta_range(start="0D", periods=4, freq="D")
    full = _build_full_index(index=idx, index_start=idx[0], index_end=idx[-1])
    assert isinstance(full, pd.TimedeltaIndex)
    assert len(full) == len(idx)


def test_build_full_index_rangeindex():
    idx = pd.RangeIndex(start=0, stop=10, step=2)  # 0,2,4,6,8
    full = _build_full_index(index=idx, index_start=0, index_end=8)
    assert isinstance(full, pd.RangeIndex)
    assert list(full) == [0, 2, 4, 6, 8]


def test_build_full_index_int_like():
    idx = pd.Index([0, 2, 4, 6])
    full = _build_full_index(index=idx, index_start=0, index_end=6)
    assert list(full) == [0, 2, 4, 6]


def test_build_full_index_numeric_fallback():
    full = _build_full_index(index_start=1, index_end=3)
    assert list(full) == [1, 2, 3]


def test_build_full_index_categorical():
    idx = pd.CategoricalIndex(["a", "b", "c", "d"], ordered=True)
    # slice b..c
    full = _build_full_index(index=idx, index_start="b", index_end="c")
    assert list(full) == ["b", "c"]


def test_build_full_index_raises_on_multiindex():
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]])
    with pytest.raises(NotImplementedError):
        _build_full_index(index=mi, index_start=mi[0], index_end=mi[-1])
