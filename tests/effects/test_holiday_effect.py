"""Tests for holiday effects."""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import pytest
from datetime import datetime

from prophetverse.effects.holiday_effect import FourierHolidayEffect
from prophetverse.effects.linear import LinearEffect # Required for isinstance checks or direct manipulation if needed

# pylint: disable=redefined-outer-name

@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    date_rng = pd.date_range(start='2024-06-28', end='2024-07-05', freq='H')
    y = pd.DataFrame(index=date_rng, data={'y': np.ones(len(date_rng))})
    X = pd.DataFrame(index=date_rng)
    return y, X

@pytest.fixture
def sample_holidays_df():
    """Create a sample holidays DataFrame."""
    holidays = pd.DataFrame({
        'holiday': ['HolidayA', 'HolidayB', 'HolidayC'],
        'ds': [datetime(2024, 7, 1), datetime(2024, 7, 1), datetime(2024, 7, 3)],
        'lower_window': [0, 0, 0],
        'upper_window': [0, 0, 0],
    })
    return holidays

class TestFourierHolidayEffect:
    """Tests for the FourierHolidayEffect class."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning") # For sktime fourier features
    def test_multiple_holidays_accumulation(self, sample_time_series_data, sample_holidays_df):
        """Test correct accumulation of effects for multiple holidays on the same day."""
        y, X = sample_time_series_data
        holidays_df = sample_holidays_df

        fourier_terms = 2
        effect = FourierHolidayEffect(
            holidays=holidays_df,
            fourier_terms=fourier_terms,
            granularity='h',
            periods_per_day=24, # Explicitly set
            effect_mode='additive',
            prior_scale=0.1,
            include_day_of_week=False # Simplifies holiday names to HolidayName_offset
        )

        # Fit the effect
        effect.fit(y=y, X=X)

        # Define forecast horizon
        fh_dates = pd.date_range(start='2024-07-01', end='2024-07-03 23:00:00', freq='H')
        X_fh = pd.DataFrame(index=fh_dates)
        
        # Transform data
        # Note: _transform is an internal method, but we call transform() which calls it.
        transformed_data = effect.transform(X=X_fh, fh=fh_dates)

        # Manually set beta coefficients for HolidayA_0 and HolidayB_0
        # Holiday names are constructed as {holiday_name}_{offset} when include_day_of_week=False
        holiday_a_name_internal = "HolidayA_0"
        holiday_b_name_internal = "HolidayB_0"
        holiday_c_name_internal = "HolidayC_0" # For the holiday on 2024-07-03

        num_betas = 2 * fourier_terms # Each term has sin and cos

        beta_a_values = jnp.array([0.5] * num_betas)
        beta_b_values = jnp.array([0.3] * num_betas)
        beta_c_values = jnp.array([0.1] * num_betas) # For HolidayC

        if holiday_a_name_internal in effect.holiday_linear_effects_:
            linear_effect_a = effect.holiday_linear_effects_[holiday_a_name_internal]
            linear_effect_a.beta_ = beta_a_values
        else:
            pytest.fail(f"'{holiday_a_name_internal}' not found in holiday_linear_effects_")

        if holiday_b_name_internal in effect.holiday_linear_effects_:
            linear_effect_b = effect.holiday_linear_effects_[holiday_b_name_internal]
            linear_effect_b.beta_ = beta_b_values
        else:
            pytest.fail(f"'{holiday_b_name_internal}' not found in holiday_linear_effects_")
        
        if holiday_c_name_internal in effect.holiday_linear_effects_:
            linear_effect_c = effect.holiday_linear_effects_[holiday_c_name_internal]
            linear_effect_c.beta_ = beta_c_values
        # If HolidayC is not in X_fh range for training, it might not be in holiday_linear_effects_
        # This is fine, it just means it won't contribute to the effect.
        # The test setup ensures HolidayC (2024-07-03) is in fh_dates.

        # Predict effect
        predicted_effect_output = effect.predict(data=transformed_data, predicted_effects={})

        # Assertions
        assert predicted_effect_output.shape == (len(fh_dates), 1), "Shape of predicted effect is incorrect."

        # Manual calculation for accumulation check
        # The `transformed_data` from FourierHolidayEffect._transform contains:
        # 'holiday_data': { holiday_name_internal: data_for_linear_effect, ... }
        # where data_for_linear_effect is {'features': holiday_fourier_features_array, ...}

        # Expected effect for Holiday A on 2024-07-01
        effect_a_manual = jnp.zeros((len(fh_dates), 1))
        if holiday_a_name_internal in transformed_data['holiday_data']:
            features_a = transformed_data['holiday_data'][holiday_a_name_internal]['features']
            # features_a is (n_timesteps_in_fh, n_series, num_betas) if X_fh has multiple series
            # or (n_timesteps_in_fh, num_betas) for single series.
            # Our y is single series, so features_a should be (len(fh_dates), num_betas)
            if features_a.ndim == 3 and features_a.shape[1] == 1: # (T, N, F) -> (T, F)
                 features_a = jnp.squeeze(features_a, axis=1)
            effect_a_manual = features_a @ beta_a_values.reshape(-1, 1)
        
        # Expected effect for Holiday B on 2024-07-01
        effect_b_manual = jnp.zeros((len(fh_dates), 1))
        if holiday_b_name_internal in transformed_data['holiday_data']:
            features_b = transformed_data['holiday_data'][holiday_b_name_internal]['features']
            if features_b.ndim == 3 and features_b.shape[1] == 1: # (T, N, F) -> (T, F)
                 features_b = jnp.squeeze(features_b, axis=1)
            effect_b_manual = features_b @ beta_b_values.reshape(-1, 1)

        # Expected effect for Holiday C on 2024-07-03
        effect_c_manual = jnp.zeros((len(fh_dates), 1))
        if holiday_c_name_internal in transformed_data['holiday_data'] and \
           holiday_c_name_internal in effect.holiday_linear_effects_: # ensure model exists
            features_c = transformed_data['holiday_data'][holiday_c_name_internal]['features']
            if features_c.ndim == 3 and features_c.shape[1] == 1: # (T, N, F) -> (T, F)
                 features_c = jnp.squeeze(features_c, axis=1)
            effect_c_manual = features_c @ beta_c_values.reshape(-1, 1)


        expected_total_effect = effect_a_manual + effect_b_manual + effect_c_manual
        
        # Check accumulation on 2024-07-01
        # Timestamps for 2024-07-01
        date_idx_2024_07_01 = (fh_dates >= '2024-07-01') & (fh_dates < '2024-07-02')
        
        # For 2024-07-01, effect should be A+B
        # Holiday C is on 07-03, so effect_c_manual should be zero on 07-01
        expected_on_2024_07_01 = (effect_a_manual[date_idx_2024_07_01] + 
                                  effect_b_manual[date_idx_2024_07_01])
        predicted_on_2024_07_01 = predicted_effect_output[date_idx_2024_07_01]
        
        assert jnp.all(predicted_on_2024_07_01 != 0), "Effect should be non-zero on 2024-07-01"
        np.testing.assert_allclose(
            predicted_on_2024_07_01,
            expected_on_2024_07_01,
            rtol=1e-5,
            err_msg="Accumulation incorrect on 2024-07-01 for Holiday A & B."
        )

        # Check effect on 2024-07-02 (should be zero as no holiday active)
        date_idx_2024_07_02 = (fh_dates >= '2024-07-02') & (fh_dates < '2024-07-03')
        predicted_on_2024_07_02 = predicted_effect_output[date_idx_2024_07_02]
        np.testing.assert_allclose(
            predicted_on_2024_07_02,
            jnp.zeros_like(predicted_on_2024_07_02),
            atol=1e-6, # Using atol for checking against zero
            err_msg="Effect should be zero on 2024-07-02."
        )

        # Check effect on 2024-07-03 (should be only Holiday C)
        date_idx_2024_07_03 = (fh_dates >= '2024-07-03') & (fh_dates < '2024-07-04')
        expected_on_2024_07_03 = effect_c_manual[date_idx_2024_07_03]
        predicted_on_2024_07_03 = predicted_effect_output[date_idx_2024_07_03]

        # Only assert if Holiday C was actually modeled and its features are present
        if holiday_c_name_internal in transformed_data['holiday_data'] and \
           holiday_c_name_internal in effect.holiday_linear_effects_:
            assert jnp.all(predicted_on_2024_07_03 != 0), "Effect should be non-zero on 2024-07-03 due to HolidayC"
            np.testing.assert_allclose(
                predicted_on_2024_07_03,
                expected_on_2024_07_03,
                rtol=1e-5,
                err_msg="Effect incorrect on 2024-07-03 for Holiday C."
            )
        else: # If Holiday C wasn't modeled (e.g. not in training data range used by fit)
             np.testing.assert_allclose(
                predicted_on_2024_07_03,
                jnp.zeros_like(predicted_on_2024_07_03),
                atol=1e-6,
                err_msg="Effect should be zero on 2024-07-03 if HolidayC was not modeled."
            )

        # Overall check against the sum of manually calculated individual effects
        np.testing.assert_allclose(
            predicted_effect_output,
            expected_total_effect,
            rtol=1e-5,
            err_msg="Overall predicted effect does not match sum of manual calculations."
        )
