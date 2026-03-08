import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects import LinearEffect, LinearFourierSeasonality
from prophetverse.effects.fourier import _coerce_period


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

@pytest.fixture
def exog_data_10days():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    return pd.DataFrame(index=idx)


class _FailAsfreqPeriod(pd.Period):
    """pd.Period subclass that raises ValueError from the external asfreq call.

    pd.Period.to_timestamp() internally calls asfreq('D', 'S') with an
    explicit how='S'.  The external call from _coerce_period uses only the
    default how='E'.  We raise only in the latter case so that the fallback
    branch ``pd.Period(value.to_timestamp(), freq=index.freq)`` can still
    complete successfully.
    """

    def asfreq(self, freq, how="E"):
        if how == "E":
            raise ValueError("forced incompatible frequency")
        return super().asfreq(freq, how)


def _fit_predict(effect, exog_data):
    fh = exog_data.index.get_level_values(-1).unique()
    effect.fit(X=exog_data, y=None)
    data = effect.transform(X=exog_data, fh=fh)
    trend = jnp.ones((len(fh), 1))
    with numpyro.handlers.seed(numpyro.handlers.seed, 0):
        pred = effect.predict(data, predicted_effects={"trend": trend})
    return pred, data

def test_linear_fourier_seasonality_initialization(fourier_effect_instance):
    assert fourier_effect_instance.sp_list == [365.25]
    assert fourier_effect_instance.fourier_terms_list == [3]
    assert fourier_effect_instance.freq == "D"
    assert fourier_effect_instance.prior_scale == 1.0
    assert fourier_effect_instance.effect_mode == "additive"


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
    assert isinstance(transformed["data"], jnp.ndarray)
    assert transformed["data"].shape == fourier_transformed.shape


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


# ---------------------------------------------------------------------------
# Tests for start_period / end_period masking
# ---------------------------------------------------------------------------

def test_no_mask_when_no_bounds(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D"
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    assert "mask" not in data

def test_mask_present_with_start(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-04",
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    assert "mask" in data

def test_mask_present_with_end(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        end_period="2021-01-07",
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    assert "mask" in data

def test_mask_values_start_only(  exog_data_10days):
    start = "2021-01-04"  # 4th day (0-indexed: position 3)
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period=start,
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    mask = data["mask"]
    expected = jnp.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32
    )
    assert jnp.allclose(mask, expected), f"mask={mask}, expected={expected}"

def test_mask_values_end_only(  exog_data_10days):
    end = "2021-01-05"  # 5th day (0-indexed: position 4)
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        end_period=end,
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    mask = data["mask"]
    expected = jnp.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=jnp.float32
    )
    assert jnp.allclose(mask, expected), f"mask={mask}, expected={expected}"

def test_mask_values_start_and_end(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-04",
        end_period="2021-01-07",
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    mask = data["mask"]
    expected = jnp.array(
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=jnp.float32
    )
    assert jnp.allclose(mask, expected), f"mask={mask}, expected={expected}"

def test_prediction_zero_outside_window(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-04",
        end_period="2021-01-07",
    )
    pred, _ = _fit_predict(effect, exog_data_10days)
    # positions 0,1,2 (before start) and 7,8,9 (after end) must be 0
    outside = jnp.concatenate([pred[:3], pred[7:]])
    assert jnp.allclose(outside, 0.0), f"Expected zeros outside window, got {outside}"

def test_prediction_nonzero_inside_window(  exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-04",
        end_period="2021-01-07",
    )
    pred, _ = _fit_predict(effect, exog_data_10days)
    inside = pred[3:7]
    # With random Fourier coefficients sampled from the prior, the chance
    # that every entry is exactly zero is astronomically small.
    assert not jnp.allclose(inside, 0.0), "Expected nonzero values inside window"

def test_mask_with_period_index():
    idx = pd.period_range("2021-01", periods=12, freq="M")
    exog = pd.DataFrame(index=idx)
    effect = LinearFourierSeasonality(
        sp_list=[12], fourier_terms_list=[2], freq="M",
        start_period="2021-04",
        end_period="2021-09",
    )
    fh = idx
    effect.fit(X=exog, y=None)
    data = effect.transform(X=exog, fh=fh)
    assert "mask" in data
    mask = data["mask"]
    # months 0-2 (Jan–Mar) and 9-11 (Oct–Dec) should be 0
    expected_zeros_before = mask[:3]
    expected_zeros_after = mask[9:]
    assert jnp.allclose(expected_zeros_before, 0.0)
    assert jnp.allclose(expected_zeros_after, 0.0)
    # months 3-8 (Apr–Sep) should be 1
    expected_ones = mask[3:9]
    assert jnp.allclose(expected_ones, 1.0)

def test_start_equals_end_single_step(  exog_data_10days):
    single_day = "2021-01-05"
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period=single_day,
        end_period=single_day,
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)
    mask = data["mask"]
    assert int(mask.sum()) == 1
    assert float(mask[4]) == 1.0  # 2021-01-05 is index position 4

def test_full_range_mask_equivalent_to_no_mask(  exog_data_10days):
    base_effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D"
    )
    windowed_effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-01",
        end_period="2021-01-10",
    )
    fh = exog_data_10days.index
    base_effect.fit(X=exog_data_10days, y=None)
    windowed_effect.fit(X=exog_data_10days, y=None)

    data_base = base_effect.transform(X=exog_data_10days, fh=fh)
    data_windowed = windowed_effect.transform(X=exog_data_10days, fh=fh)

    assert "mask" not in data_base
    assert "mask" in data_windowed
    # The mask should be all-ones
    assert jnp.allclose(data_windowed["mask"], 1.0)


def test_start_after_end_raises(exog_data_10days):
    """start_period > end_period should raise ValueError, not silently zero mask."""
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-08",
        end_period="2021-01-04",
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    with pytest.raises(ValueError, match="start_period.*end_period"):
        effect.transform(X=exog_data_10days, fh=fh)


def test_predict_panel_mask_reshape(exog_data_10days):
    effect = LinearFourierSeasonality(
        sp_list=[7], fourier_terms_list=[1], freq="D",
        start_period="2021-01-04",
        end_period="2021-01-07",
    )
    fh = exog_data_10days.index
    effect.fit(X=exog_data_10days, y=None)
    data = effect.transform(X=exog_data_10days, fh=fh)

    # Promote the (T, F) data array to (1, T, F) to simulate one panel series
    arr_3d = data["data"][jnp.newaxis]          # (1, T, F)
    data_3d = {"data": arr_3d, "mask": data["mask"]}

    trend = jnp.ones((1, len(fh), 1))          # (N=1, T, 1)
    with numpyro.handlers.seed(numpyro.handlers.seed, 0):
        pred = effect.predict(data_3d, predicted_effects={"trend": trend})

    # Result should be 3-D and zeros must appear outside the window
    assert pred.ndim == 3
    assert jnp.allclose(pred[:, :3, :], 0.0), "Expected zeros before start_period"
    assert jnp.allclose(pred[:, 7:, :], 0.0), "Expected zeros after end_period"


# ---------------------------------------------------------------------------
# Tests for _coerce_period branches (coverage)
# ---------------------------------------------------------------------------

def test_coerce_period_same_freq_period():
    idx = pd.period_range("2021-01", periods=6, freq="M")
    val = pd.Period("2021-03", freq="M")
    result = _coerce_period(val, idx)
    assert result == val


def test_coerce_period_different_freq_period():
    idx = pd.period_range("2021-01", periods=6, freq="M")
    # Quarterly period; asfreq("M") should succeed
    val = pd.Period("2021Q1", freq="Q-DEC")
    result = _coerce_period(val, idx)
    assert result.freqstr == "M"


def test_coerce_period_asfreq_fallback():
    idx = pd.period_range("2021-01", periods=6, freq="M")
    val = _FailAsfreqPeriod("2021Q1", freq="Q-DEC")
    result = _coerce_period(val, idx)
    assert isinstance(result, pd.Period)
    assert result.freqstr == "M"


def test_coerce_period_timestamp_passthrough():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    ts = pd.Timestamp("2021-01-05")
    result = _coerce_period(ts, idx)
    assert result == ts


def test_coerce_period_period_with_datetime_index():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    val = pd.Period("2021-01-05", freq="D")
    result = _coerce_period(val, idx)
    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp("2021-01-05")

