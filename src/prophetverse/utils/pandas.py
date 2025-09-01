import pandas as pd
import numpy as np
from collections import Counter


def _build_full_index(index=None, index_start=None, index_end=None, inclusive="both"):
    """
    Create a full contiguous index from start to end matching the type/freq of `index`.

    Parameters
    ----------
    index : pandas Index (optional)
        Existing index to infer type/frequency/step. If None, type inferred from start.
    index_start, index_end :
        Boundaries (can be scalars or same-type elements of `index`).
    inclusive : {'both','left','right','neither'}
        Passed-through inclusion policy (works for date/period/timedelta; manually handled for others).

    Returns
    -------
    pandas.Index (appropriate subclass)
    """
    if index_start is None and index is not None:
        index_start = index[0]
    if index_end is None and index is not None:
        index_end = index[-1]
    if index_start is None or index_end is None:
        raise ValueError("Must provide start/end or an index to infer them.")

    # Helper to pick freq / step
    def infer_step_numeric(idx):
        diffs = np.diff(idx.astype("int64"))
        if len(diffs) == 0:
            return 1
        # mode of diffs
        step = Counter(diffs).most_common(1)[0][0]
        return int(step)

    if isinstance(index, pd.DatetimeIndex) or (
        index is None and isinstance(index_start, (pd.Timestamp, str))
    ):
        start = pd.Timestamp(index_start)
        end = pd.Timestamp(index_end)
        if index is not None:
            freq = index.freq or pd.infer_freq(index)
        else:
            freq = None
        if freq is None and index is not None and len(index) > 2:
            # fallback: most common delta
            deltas = np.diff(index.view("int64"))
            if len(deltas):
                mode_delta = Counter(deltas).most_common(1)[0][0]
                freq = pd.tseries.frequencies.to_offset(mode_delta)
        return pd.date_range(start=start, end=end, freq=freq, inclusive=inclusive)

    if isinstance(index, pd.PeriodIndex) or (
        index is None and isinstance(index_start, pd.Period)
    ):
        start = pd.Period(
            index_start, freq=index.freq if index is not None else index_start.freq
        )
        end = pd.Period(index_end, freq=start.freq)
        # Older pandas versions lack inclusive= for period_range; build full then slice
        try:
            rng = pd.period_range(
                start=start, end=end, freq=start.freq, inclusive=inclusive
            )
        except TypeError:
            rng = pd.period_range(start=start, end=end, freq=start.freq)
            if inclusive == "neither":
                rng = rng[1:-1]
            elif inclusive == "left":
                rng = rng[:-1]
            elif inclusive == "right":
                rng = rng[1:]
        return rng

    if isinstance(index, pd.TimedeltaIndex) or (
        index is None and isinstance(index_start, pd.Timedelta)
    ):
        start = pd.Timedelta(index_start)
        end = pd.Timedelta(index_end)
        freq = index.freq if index is not None else None
        if freq is None and index is not None and len(index) > 2:
            deltas = np.diff(index.view("int64"))
            if len(deltas):
                mode_delta = Counter(deltas).most_common(1)[0][0]
                freq = pd.tseries.frequencies.to_offset(mode_delta)
        try:
            rng = pd.timedelta_range(
                start=start, end=end, freq=freq, inclusive=inclusive
            )
        except TypeError:
            rng = pd.timedelta_range(start=start, end=end, freq=freq)
            if inclusive == "neither":
                rng = rng[1:-1]
            elif inclusive == "left":
                rng = rng[:-1]
            elif inclusive == "right":
                rng = rng[1:]
        return rng

    if isinstance(index, pd.RangeIndex) or (
        index is None and isinstance(index_start, (int, np.integer))
    ):
        step = index.step if isinstance(index, pd.RangeIndex) else 1
        start_i = int(index_start)
        end_i = int(index_end)
        include_start = inclusive in ("both", "left")
        include_end = inclusive in ("both", "right")
        if not include_start:
            start_i += step
        stop = end_i + step if include_end else end_i
        if stop <= start_i:  # empty or invalid; return empty RangeIndex
            return pd.RangeIndex(start=0, stop=0, step=1)
        return pd.RangeIndex(start=start_i, stop=stop, step=step)

    # Generic integer-like index (e.g., plain Index of ints)
    if (index is not None and index.dtype.kind in ("i", "u")) or (
        index is None and isinstance(index_start, (int, np.integer))
    ):
        if index is not None and len(index) > 1:
            step = infer_step_numeric(np.asarray(index))
        else:
            step = 1
        start_i = int(index_start)
        end_i = int(index_end)
        include_start = inclusive in ("both", "left")
        include_end = inclusive in ("both", "right")
        if include_start and include_end:
            values = np.arange(start_i, end_i + step, step)
        elif include_start and not include_end:
            values = np.arange(start_i, end_i, step)
        elif not include_start and include_end:
            values = np.arange(start_i + step, end_i + step, step)
        else:  # neither
            values = np.arange(start_i + step, end_i, step)
        return pd.Index(values)

    if isinstance(index, pd.CategoricalIndex):
        # Cannot interpolate unseen categories; return slice
        cats = index.categories
        mask = (index >= index_start) & (index <= index_end)
        return pd.CategoricalIndex(index[mask], categories=cats, ordered=index.ordered)

    if isinstance(index, pd.MultiIndex):
        raise NotImplementedError(
            "Span generation for MultiIndex requires domain-specific rules."
        )

    # Fallback: try to build an Index from python range if numeric else raise.
    try:
        start_i, end_i = int(index_start), int(index_end)
        return pd.Index(range(start_i, end_i + 1))
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"Unsupported index type: {type(index)}") from e
