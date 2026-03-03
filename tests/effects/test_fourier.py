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

@pytest.fixture
def exog_data_10days():
    """10-day DatetimeIndex fixture reused by masking tests."""
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    return pd.DataFrame(index=idx)


def _fit_predict(effect, exog_data):
    """Helper: fit → transform → predict, return (prediction, data)."""
    fh = exog_data.index.get_level_values(-1).unique()
    effect.fit(X=exog_data, y=None)
    data = effect.transform(X=exog_data, fh=fh)
    trend = jnp.ones((len(fh), 1))
    with numpyro.handlers.seed(numpyro.handlers.seed, 0):
        pred = effect.predict(data, predicted_effects={"trend": trend})
    return pred, data


class TestLinearFourierSeasonalityMask:
    """Tests for the optional start_period / end_period windowing feature."""

    def test_no_mask_when_no_bounds(self, exog_data_10days):
        """No 'mask' key should appear when neither bound is supplied."""
        effect = LinearFourierSeasonality(
            sp_list=[7], fourier_terms_list=[1], freq="D"
        )
        fh = exog_data_10days.index
        effect.fit(X=exog_data_10days, y=None)
        data = effect.transform(X=exog_data_10days, fh=fh)
        assert "mask" not in data

    def test_mask_present_with_start(self, exog_data_10days):
        """'mask' key should be present when start_period is set."""
        effect = LinearFourierSeasonality(
            sp_list=[7], fourier_terms_list=[1], freq="D",
            start_period="2021-01-04",
        )
        fh = exog_data_10days.index
        effect.fit(X=exog_data_10days, y=None)
        data = effect.transform(X=exog_data_10days, fh=fh)
        assert "mask" in data

    def test_mask_present_with_end(self, exog_data_10days):
        """'mask' key should be present when end_period is set."""
        effect = LinearFourierSeasonality(
            sp_list=[7], fourier_terms_list=[1], freq="D",
            end_period="2021-01-07",
        )
        fh = exog_data_10days.index
        effect.fit(X=exog_data_10days, y=None)
        data = effect.transform(X=exog_data_10days, fh=fh)
        assert "mask" in data

    def test_mask_values_start_only(self, exog_data_10days):
        """Mask should be 0 before start_period and 1 from start_period onward."""
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

    def test_mask_values_end_only(self, exog_data_10days):
        """Mask should be 1 up to and including end_period, then 0."""
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

    def test_mask_values_start_and_end(self, exog_data_10days):
        """Mask should be 1 only within [start_period, end_period]."""
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

    def test_prediction_zero_outside_window(self, exog_data_10days):
        """Prediction output must be exactly zero for time-steps outside the window."""
        effect = LinearFourierSeasonality(
            sp_list=[7], fourier_terms_list=[1], freq="D",
            start_period="2021-01-04",
            end_period="2021-01-07",
        )
        pred, _ = _fit_predict(effect, exog_data_10days)
        # positions 0,1,2 (before start) and 7,8,9 (after end) must be 0
        outside = jnp.concatenate([pred[:3], pred[7:]])
        assert jnp.allclose(outside, 0.0), f"Expected zeros outside window, got {outside}"

    def test_prediction_nonzero_inside_window(self, exog_data_10days):
        """At least some predictions inside the window should be nonzero."""
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

    def test_mask_with_period_index(self):
        """Masking should also work when the index is a PeriodIndex."""
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

    def test_start_equals_end_single_step(self, exog_data_10days):
        """start_period == end_period should produce a mask with exactly one 1."""
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

    def test_full_range_mask_equivalent_to_no_mask(self, exog_data_10days):
        """Supplying the full date range should produce predictions equal to no mask."""
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

