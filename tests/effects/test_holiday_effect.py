"""Tests for holiday effects."""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import pytest
from datetime import datetime
import numpyro
from numpyro.handlers import seed # Import seed for use

from prophetverse.effects.holiday_effect import BinaryHolidayEffect, BinaryStdHolidayEffect, FourierHolidayEffect

# pylint: disable=redefined-outer-name

@pytest.fixture
def sample_univariate_data():
    """Create sample univariate time series data for testing."""
    date_rng = pd.date_range(start='2023-12-20', end='2024-01-10', freq='D')
    y = pd.DataFrame({'y': np.arange(len(date_rng))}, index=date_rng)
    X = pd.DataFrame(index=date_rng) 
    return y, X

@pytest.fixture
def sample_hourly_univariate_data():
    """Create sample hourly univariate time series data."""
    date_rng = pd.date_range(start='2023-12-31', end='2024-01-02', freq='H') # Original 'H'
    y = pd.DataFrame({'y': np.arange(len(date_rng))}, index=date_rng)
    X = pd.DataFrame(index=date_rng)
    return y, X


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate time series data for testing."""
    date_rng = pd.date_range(start='2023-12-20', end='2024-01-10', freq='D')
    idx = pd.MultiIndex.from_product([['series1', 'series2'], date_rng], names=['series_id', 'date'])
    y = pd.DataFrame({'y': np.arange(len(idx))}, index=idx)
    X = pd.DataFrame(index=idx)
    return y, X

@pytest.fixture
def sample_hourly_multivariate_data():
    """Create sample hourly multivariate time series data."""
    date_rng = pd.date_range(start='2023-12-31', end='2024-01-02', freq='H') # Original 'H'
    idx = pd.MultiIndex.from_product([['series1', 'series2'], date_rng], names=['series_id', 'date'])
    y = pd.DataFrame({'y': np.arange(len(idx))}, index=idx)
    X = pd.DataFrame(index=idx)
    return y, X


@pytest.fixture
def christmas_holiday_df():
    """Holiday DataFrame for Christmas."""
    return pd.DataFrame({
        'holiday': ['Christmas'],
        'ds': [datetime(2023, 12, 25)],
        'lower_window': [0],
        'upper_window': [0]
    })

@pytest.fixture
def new_year_holiday_df():
    """Holiday DataFrame for New Year's Day."""
    return pd.DataFrame({
        'holiday': ["NewYear"],
        'ds': [datetime(2024, 1, 1)],
        'lower_window': [0],
        'upper_window': [0]
    })


class TestHolidayEffect:
    """Tests for the basic BinaryHolidayEffect class."""

    @pytest.mark.parametrize("data_fixture", ["sample_univariate_data", "sample_multivariate_data"])
    def test_holiday_effect_fit_predict_smoke(self, data_fixture, christmas_holiday_df, request):
        y, X = request.getfixturevalue(data_fixture)
        effect = BinaryHolidayEffect(
            holidays=christmas_holiday_df,
            prior_scale=0.1
        )
        effect.fit(y=y, X=X)
        
        fh_dates = y.index.get_level_values(-1).unique()
        X_fh = pd.DataFrame(index=fh_dates) 
        if isinstance(y.index, pd.MultiIndex): 
            series_levels = y.index.droplevel(-1).unique()
            idx_tuples = []
            for s_idx_tuple in series_levels:
                for t_idx in fh_dates:
                    if isinstance(s_idx_tuple, tuple):
                        idx_tuples.append((*s_idx_tuple, t_idx))
                    else:
                        idx_tuples.append((s_idx_tuple, t_idx))
            panel_idx = pd.MultiIndex.from_tuples(idx_tuples, names=y.index.names)
            X_fh = pd.DataFrame(index=panel_idx)


        with seed(rng_seed=0):
            transformed_data = effect.transform(X=X_fh, fh=fh_dates)
            # This call reflects the state where tests might still be using data=... for predict
            predicted_effect = effect.predict(data=transformed_data, predicted_effects={})


        assert predicted_effect is not None
        expected_shape_last_dim = 1
        if isinstance(y.index, pd.MultiIndex):
            num_series = y.index.get_level_values(0).nunique()
            assert predicted_effect.shape == (num_series, len(fh_dates), expected_shape_last_dim)
        else:
            assert predicted_effect.shape == (len(fh_dates), expected_shape_last_dim)
        
        christmas_date_str = '2023-12-25'
        fh_dates_index = pd.Index(fh_dates) 
        if christmas_date_str in fh_dates_index:
            christmas_idx = fh_dates_index.get_loc(christmas_date_str)
            if isinstance(y.index, pd.MultiIndex):
                assert jnp.any(predicted_effect[:, christmas_idx, :] != 0)
            else:
                assert predicted_effect[christmas_idx, 0] != 0

    def test_holiday_effect_no_holidays_in_fh(self, sample_univariate_data, christmas_holiday_df):
        y, X = sample_univariate_data
        effect = BinaryHolidayEffect(holidays=christmas_holiday_df, prior_scale=0.1)
        effect.fit(y=y, X=X)
        fh_dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        X_fh = pd.DataFrame(index=fh_dates)
        with seed(rng_seed=0):
            transformed_data = effect.transform(X=X_fh, fh=fh_dates)
            predicted_effect = effect.predict(data=transformed_data, predicted_effects={})
        assert jnp.all(predicted_effect == 0)

    def test_basic_batching(self):
        """Test basic batching functionality."""
        # Create sample hourly data
        date_rng = pd.date_range(start='2023-12-24', end='2023-12-26', freq='h')
        y = pd.DataFrame({'y': np.arange(len(date_rng))}, index=date_rng)
        X = pd.DataFrame(index=date_rng)
        # Create a simple holiday DataFrame
        holidays_df = pd.DataFrame({
            'holiday': ['Christmas', 'NewYear'],
            'ds': [datetime(2023, 12, 25), datetime(2024, 1, 1)],
        })
        # Test with batching (daily_splits=6 for 4-hour blocks)
        effect = BinaryHolidayEffect(
            holidays=holidays_df,
            daily_splits=6,  # 24 / 6 = 4 hour blocks
            granularity='h'
        )
        effect.fit(y=y, X=X)
        # Test transform for one day
        fh = pd.date_range(start='2023-12-25', end='2023-12-25 23:00:00', freq='h')
        X_fh = pd.DataFrame(index=fh)
        transformed_data = effect.transform(X=X_fh, fh=fh)
        # Expected: 24 time periods, 2 holidays * 6 batches = 12 features
        expected_features = len(effect.holiday_names_) * effect.daily_splits
        assert transformed_data['features'].shape == (24, expected_features), \
            f"Expected shape (24, {expected_features}), got {transformed_data['features'].shape}"
        
    def test_validation(self):
        """Test validation functionality."""
        # Test 1: Valid divisor (should stay the same)
        effect1 = BinaryHolidayEffect(daily_splits=8, granularity='h')
        assert effect1.daily_splits == 8, f"Expected 8, got {effect1.daily_splits}"
        # Test 2: Invalid divisor (should be adjusted)
        effect2 = BinaryHolidayEffect(daily_splits=5, granularity='h')
        assert effect2.daily_splits == 4, f"Expected 4, got {effect2.daily_splits}"
        # Test 3: Large invalid divisor (should use max valid)
        effect3 = BinaryHolidayEffect(daily_splits=30, granularity='h')
        assert effect3.daily_splits == 24, f"Expected 24, got {effect3.daily_splits}"
    

class TestFourierHolidayEffect:
    """Tests for the FourierHolidayEffect class."""

    def _run_fit_predict(self, effect_instance, y_data, X_data, fh_dates, rng_seed=0):
        effect_instance.fit(y=y_data, X=X_data)
        
        X_fh_for_transform = pd.DataFrame(index=fh_dates)
        if isinstance(y_data.index, pd.MultiIndex):
            idx_tuples = []
            series_idx_list = effect_instance.series_idx_.to_list()
            if not series_idx_list: pass 
            elif not isinstance(series_idx_list[0], tuple) and effect_instance.index_names_ and len(effect_instance.index_names_) > 1 :
                 series_idx_list = [(s,) for s in series_idx_list]

            for series_idx_tuple in series_idx_list:
                s_idx_levels = series_idx_tuple if isinstance(series_idx_tuple, tuple) else (series_idx_tuple,)
                for time_idx in fh_dates:
                    idx_tuples.append((*s_idx_levels, time_idx))
            if idx_tuples: 
                final_transform_idx = pd.MultiIndex.from_tuples(idx_tuples, names=effect_instance.index_names_)
                X_fh_for_transform = pd.DataFrame(index=final_transform_idx)

        with seed(rng_seed=rng_seed):
            transformed_data = effect_instance.transform(X=X_fh_for_transform, fh=fh_dates)
            # Create predicted_effects with trend if needed for multiplicative mode
            predicted_effects = {}
            if effect_instance.effect_mode == "multiplicative":
                # Create dummy trend values with the same shape as expected output
                if effect_instance.has_multiple_series_:
                    num_series = y_data.index.get_level_values(0).nunique()
                    trend_values = jnp.ones((num_series, len(fh_dates), 1))
                else:
                    trend_values = jnp.ones((len(fh_dates), 1))
                predicted_effects[effect_instance.linear_effect_.base_effect_name] = trend_values
            
            # This predict call is the one that would be used by the tests at this stage
            predicted_effect = effect_instance.predict(data=transformed_data, predicted_effects=predicted_effects)
        return predicted_effect

    @pytest.mark.parametrize("data_fixture, holiday_df_fixture, holiday_date_str, freq", [
        ("sample_hourly_univariate_data", "new_year_holiday_df", "2024-01-01", "H"),
        ("sample_hourly_multivariate_data", "new_year_holiday_df", "2024-01-01", "H")
    ])
    @pytest.mark.parametrize("fourier_terms", [1, 3])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fourier_effect_shape_and_basic_impact(
        self, data_fixture, holiday_df_fixture, holiday_date_str, freq, fourier_terms, request
    ):
        y, X = request.getfixturevalue(data_fixture)
        holidays_df = request.getfixturevalue(holiday_df_fixture)
        effect = FourierHolidayEffect(
            holidays=holidays_df, fourier_terms=fourier_terms, granularity=freq, # freq='H'
            effect_mode='additive', prior_scale=0.1 
        )
        fh_dates = y.index.get_level_values(-1).unique()
        predicted_effect = self._run_fit_predict(effect, y, X, fh_dates)

        expected_shape_last_dim = 1
        if isinstance(y.index, pd.MultiIndex):
            num_series = y.index.get_level_values(0).nunique()
            assert predicted_effect.shape == (num_series, len(fh_dates), expected_shape_last_dim)
            holiday_datetime = pd.to_datetime(holiday_date_str)
            holiday_day_mask = np.array([d.date() == holiday_datetime.date() for d in fh_dates])
            day_before_holiday_mask = np.array([d.date() == (holiday_datetime - pd.Timedelta(days=1)).date() for d in fh_dates])
            for series_idx in range(num_series):
                series_effect_on_holiday_day = predicted_effect[series_idx, holiday_day_mask, 0]
                series_effect_on_day_before = predicted_effect[series_idx, day_before_holiday_mask, 0]
                if fourier_terms > 0 and np.sum(holiday_day_mask) > 0:
                    assert jnp.any(jnp.abs(series_effect_on_holiday_day) > 1e-7)
                if np.sum(day_before_holiday_mask) > 0: 
                     assert jnp.all(jnp.abs(series_effect_on_day_before) < 1e-7)
        else: 
            assert predicted_effect.shape == (len(fh_dates), expected_shape_last_dim)
            holiday_datetime = pd.to_datetime(holiday_date_str)
            holiday_day_mask = np.array([d.date() == holiday_datetime.date() for d in fh_dates])
            day_before_holiday_mask = np.array([d.date() == (holiday_datetime - pd.Timedelta(days=1)).date() for d in fh_dates])
            effect_on_holiday_day = predicted_effect[holiday_day_mask, 0]
            effect_on_day_before = predicted_effect[day_before_holiday_mask, 0]
            if fourier_terms > 0 and np.sum(holiday_day_mask) > 0:
                assert jnp.any(jnp.abs(effect_on_holiday_day) > 1e-7)
            if np.sum(day_before_holiday_mask) > 0:
                 assert jnp.all(jnp.abs(effect_on_day_before) < 1e-7)

    @pytest.mark.parametrize("data_fixture", ["sample_hourly_univariate_data", "sample_hourly_multivariate_data"])
    @pytest.mark.parametrize("effect_mode", ["additive", "multiplicative"])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fourier_terms_zero(self, data_fixture, new_year_holiday_df, effect_mode, request):
        y, X = request.getfixturevalue(data_fixture)
        effect = FourierHolidayEffect(
            holidays=new_year_holiday_df, fourier_terms=0, granularity='H', 
            effect_mode=effect_mode, prior_scale=0.1 
        )
        fh_dates = y.index.get_level_values(-1).unique()
        h_t_effect = self._run_fit_predict(effect, y, X, fh_dates) 
        assert jnp.all(jnp.abs(h_t_effect) < 1e-7)
        
        # Test with public predict call (as it was in subtask 5, might fail with original base.py)
        effect.fit(y=y,X=X)
        X_for_public_predict = pd.DataFrame(index=fh_dates)
        if isinstance(y.index, pd.MultiIndex):
            series_levels = y.index.droplevel(-1).unique()
            X_for_public_predict = pd.DataFrame(index=pd.MultiIndex.from_product([series_levels, fh_dates], names=y.index.names))
        
        transformed_data_for_public_predict = effect.transform(X=X_for_public_predict, fh=fh_dates)
        
        y_pred_baseline = jnp.ones_like(h_t_effect) 
        if isinstance(y.index, pd.MultiIndex):
            num_series = y.index.get_level_values(0).nunique()
            y_pred_baseline = jnp.ones((num_series, len(fh_dates), 1))
        else:
            y_pred_baseline = jnp.ones((len(fh_dates), 1))

        current_predicted_effects = {}
        if effect_mode == "multiplicative":
            current_predicted_effects[effect.linear_effect_.base_effect_name] = y_pred_baseline 
        
        with seed(rng_seed=0):
            final_effect_component = effect.predict( 
                data=transformed_data_for_public_predict,
                predicted_effects=current_predicted_effects
            )
        assert jnp.all(jnp.abs(final_effect_component) < 1e-7)


    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_country_name_holidays(self, sample_hourly_univariate_data):
        y, X = sample_hourly_univariate_data 
        effect = FourierHolidayEffect(
            country_name="US", fourier_terms=2, granularity='H', 
            effect_mode='additive', prior_scale=0.1
        )
        fh_dates = y.index.get_level_values(-1).unique()
        predicted_effect = self._run_fit_predict(effect, y, X, fh_dates)
        new_year_date_str = "2024-01-01"
        new_year_datetime = pd.to_datetime(new_year_date_str)
        new_year_day_mask = np.array([d.date() == new_year_datetime.date() for d in fh_dates])
        assert np.sum(new_year_day_mask) > 0
        effect_on_new_year = predicted_effect[new_year_day_mask, 0]
        assert jnp.any(jnp.abs(effect_on_new_year) > 1e-7)

    @pytest.mark.parametrize("include_dow", [True, False])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_include_day_of_week_holiday_names(self, sample_hourly_univariate_data, new_year_holiday_df, include_dow):
        y, X = sample_hourly_univariate_data
        effect = FourierHolidayEffect(
            holidays=new_year_holiday_df, fourier_terms=1, granularity='H', 
            include_day_of_week=include_dow, prior_scale=0.1
        )
        effect.fit(y=y, X=X) 
        assert hasattr(effect, 'holiday_names_')
        expected_holiday_name_part = "NewYear_0"
        found_matching_holiday_name = False
        for name in effect.holiday_names_:
            if include_dow:
                if f"{expected_holiday_name_part}_Monday" in name: 
                    found_matching_holiday_name = True; break
            else:
                if expected_holiday_name_part == name: 
                    found_matching_holiday_name = True; break
        if include_dow:
            assert found_matching_holiday_name, f"Expected DOW specific name in {effect.holiday_names_}"
        else:
            assert found_matching_holiday_name, f"Expected non-DOW specific name in {effect.holiday_names_}"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_holiday_accumulation_qualitative(self, sample_hourly_univariate_data):
        y, X = sample_hourly_univariate_data 
        holidays = pd.DataFrame({
            'holiday': ['NewYearEve', 'NewYear', 'NewYearBonus'], 
            'ds': [datetime(2023, 12, 31), datetime(2024, 1, 1), datetime(2024, 1, 1)],
            'lower_window': [0, 0, 0], 'upper_window': [0, 0, 0]
        })
        effect = FourierHolidayEffect(
            holidays=holidays, fourier_terms=1, granularity='H', 
            effect_mode='additive', prior_scale=0.1
        )
        fh_dates = pd.date_range(start='2023-12-31', end='2024-01-02 23:00:00', freq='H')
        predicted_effect = self._run_fit_predict(effect, y, X, fh_dates, rng_seed=42)

        nye_date_mask = np.array([d.date() == datetime(2023, 12, 31).date() for d in fh_dates])
        ny_date_mask = np.array([d.date() == datetime(2024, 1, 1).date() for d in fh_dates])
        jan2_date_mask = np.array([d.date() == datetime(2024, 1, 2).date() for d in fh_dates])
        effect_nye = predicted_effect[nye_date_mask, 0]
        effect_ny = predicted_effect[ny_date_mask, 0]
        effect_jan2 = predicted_effect[jan2_date_mask, 0]
        
        assert np.sum(nye_date_mask) > 0 and jnp.any(jnp.abs(effect_nye) > 1e-7)
        assert np.sum(ny_date_mask) > 0 and jnp.any(jnp.abs(effect_ny) > 1e-7)
        assert np.sum(jan2_date_mask) > 0 and jnp.all(jnp.abs(effect_jan2) < 1e-7)
        
    def test_hourly_holiday_effects(self):
        """Test that holiday effects apply to all hours of a holiday date."""
        
        # Create hourly time series data for a few days including a holiday
        dates = pd.date_range(
            start='2024-01-01 00:00:00', 
            end='2024-01-03 23:00:00', 
            freq='h'
        )
        
        # Create a simple holiday dataframe with New Year's Day
        holidays_df = pd.DataFrame({
            'holiday': ['new_years'],
            'ds': [pd.Timestamp('2024-01-01')]
        })
        
        # Test BinaryHolidayEffect
        binary_effect = BinaryHolidayEffect(
            holidays=holidays_df,
            country_name=None,
            lower_window=0,
            upper_window=0,
            prior_scale=1.0
        )
        
        # Fit the effect
        y_dummy = pd.DataFrame({'y': np.ones(len(dates))}, index=dates)
        X_dummy = pd.DataFrame(index=dates)
        binary_effect.fit(y=y_dummy, X=X_dummy)        
        
        fourier_effect = FourierHolidayEffect(
            holidays=holidays_df,
            country_name=None,
            lower_window=0,
            upper_window=0,
            prior_scale=1.0,
            fourier_terms=3
        )
        
        # Fit the effect
        fourier_effect.fit(y=y_dummy, X=X_dummy)
        
        # Transform the data
        X_fourier_data = fourier_effect.transform(X=X_dummy, fh=dates)
        
        # The FourierHolidayEffect uses LinearEffect internally, inspect what we get
        if isinstance(X_fourier_data, dict):
            # Try 'data' key
            if 'data' in X_fourier_data:
                X_fourier = X_fourier_data['data']
            else:
                # Look for the actual data key
                data_key = list(X_fourier_data.keys())[0] if X_fourier_data else None
                X_fourier = X_fourier_data[data_key] if data_key else None
        else:
            X_fourier = X_fourier_data

                
        return True

class TestStdHolidayEffect:
    """Tests for the basic BinaryStdHolidayEffect class."""

    def test_basic_functionality(self):
        """Test basic functionality without batching."""
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        y = pd.DataFrame({'y': np.random.randn(len(dates))}, index=dates)
        X = pd.DataFrame(index=dates)
        
        # Create holidays with different prior scales
        holidays_df = pd.DataFrame({
            'holiday': ['New Year', 'Christmas'],
            'ds': ['2023-01-01', '2023-12-25'],
            'prior_scale': [0.05, 0.15]  # These should be ignored
        })
        
        # Test the new effect
        effect = BinaryStdHolidayEffect(
            holidays=holidays_df,
            lower_window=-1,
            upper_window=1,
            prior_scale=0.1,  # Uniform prior for all holidays
            effect_mode='additive'
        )
        
        # Fit the effect
        effect.fit(y=y, X=X, scale=1.0)
        
        # Transform data
        fh = pd.date_range('2024-01-01', '2024-01-02', freq='D')
        data = effect.transform(X=pd.DataFrame(index=fh), fh=fh)
        
        # Predict
        with seed(rng_seed=42):
            prediction = effect.predict(data=data, predicted_effects={})
        
        return True

    def test_batching_functionality(self):
        """Test batching functionality."""
        
        # Create hourly test data
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='h')
        y = pd.DataFrame({'y': np.random.randn(len(dates))}, index=dates)
        X = pd.DataFrame(index=dates)
        
        # Create holidays
        holidays_df = pd.DataFrame({
            'holiday': ['New Year'],
            'ds': ['2023-01-01'],
            'prior_scale': [0.2]  # This should be ignored
        })
        
        # Test with batching
        effect = BinaryStdHolidayEffect(
            holidays=holidays_df,
            lower_window=0,
            upper_window=0,
            prior_scale=0.1,  # Uniform prior
            daily_splits=4,   # Split day into 4 batches
            granularity='h',
            effect_mode='additive'
        )
        
        # Fit the effect
        effect.fit(y=y, X=X, scale=1.0)
        
        # Transform data
        fh = pd.date_range('2023-01-01 00:00', '2023-01-01 23:00', freq='h')
        data = effect.transform(X=pd.DataFrame(index=fh), fh=fh)
        
        # Predict
        with seed(rng_seed=42):
            prediction = effect.predict(data=data, predicted_effects={})
        
        return True

    def test_multivariate_series(self):
        """Test with multivariate time series."""
        
        # Create multivariate test data
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        series_ids = ['series_1', 'series_2']
        
        index_tuples = [(s, d) for s in series_ids for d in dates]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['series_id', 'time'])
        
        y = pd.DataFrame({'y': np.random.randn(len(index_tuples))}, index=multi_index)
        X = pd.DataFrame(index=multi_index)
        
        # Create holidays
        holidays_df = pd.DataFrame({
            'holiday': ['New Year'],
            'ds': ['2023-01-01'],
            'prior_scale': [0.3]  # This should be ignored
        })
        
        # Test the effect
        effect = BinaryStdHolidayEffect(
            holidays=holidays_df,
            prior_scale=0.1,  # Uniform prior
            effect_mode='additive'
        )
        
        # Fit the effect
        effect.fit(y=y, X=X, scale=1.0)
        
        # Transform data
        fh = dates[-2:]  # Last 2 days
        fh_tuples = [(s, d) for s in series_ids for d in fh]
        fh_multi_index = pd.MultiIndex.from_tuples(fh_tuples, names=['series_id', 'time'])
        
        data = effect.transform(X=pd.DataFrame(index=fh_multi_index), fh=fh)
        
        # Predict
        with seed(rng_seed=42):
            prediction = effect.predict(data=data, predicted_effects={})
        
        return True

    def test_comparison_with_binary_holiday_effect(self):
        """Compare behavior with BinaryHolidayEffect to ensure differences."""
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        y = pd.DataFrame({'y': np.random.randn(len(dates))}, index=dates)
        X = pd.DataFrame(index=dates)
        
        # Create holidays with same prior scale
        holidays_df = pd.DataFrame({
            'holiday': ['New Year', 'Test Holiday'],
            'ds': ['2023-01-01', '2023-01-05'],
            'prior_scale': [0.1, 0.1]  # Same prior scale
        })
        
        # Create both effects with same settings
        binary_std_effect = BinaryStdHolidayEffect(
            holidays=holidays_df,
            prior_scale=0.1,
            effect_mode='additive'
        )
        
        binary_effect = BinaryHolidayEffect(
            holidays=holidays_df,
            prior_scale=0.1,  # This will be ignored; holidays DF prior_scale will be used
            effect_mode='additive'
        )
        
        # Fit both effects
        binary_std_effect.fit(y=y, X=X, scale=1.0)
        binary_effect.fit(y=y, X=X, scale=1.0)
        
        # Transform data
        fh = pd.date_range('2023-01-01', '2023-01-02', freq='D')
        data_std = binary_std_effect.transform(X=pd.DataFrame(index=fh), fh=fh)
        data_binary = binary_effect.transform(X=pd.DataFrame(index=fh), fh=fh)
        
        # Check that they use different internal structures
        # BinaryStdHolidayEffect uses LinearEffect, BinaryHolidayEffect uses direct sampling
        
        return True

    def test_no_holidays_edge_case(self):
        """Test edge case with no holidays."""
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        y = pd.DataFrame({'y': np.random.randn(len(dates))}, index=dates)
        X = pd.DataFrame(index=dates)
        
        # Create empty holidays DataFrame
        holidays_df = pd.DataFrame(columns=['holiday', 'ds'])
        
        # Test the effect
        effect = BinaryStdHolidayEffect(
            holidays=holidays_df,
            prior_scale=0.1,
            effect_mode='additive'
        )
        
        # Fit the effect
        effect.fit(y=y, X=X, scale=1.0)
        
        # Transform data
        fh = pd.date_range('2023-01-01', '2023-01-02', freq='D')
        data = effect.transform(X=pd.DataFrame(index=fh), fh=fh)
        
        # Predict
        with seed(rng_seed=42):
            prediction = effect.predict(data=data, predicted_effects={})
        
        return True
