"""Holiday effects for time series forecasting."""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from datetime import datetime, timedelta

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseAdditiveOrMultiplicativeEffect
from prophetverse.utils.frame_to_array import series_to_tensor
from prophetverse.effects.linear import LinearEffect
from sktime.transformations.series.fourier import FourierFeatures
from prophetverse.sktime._expand_column_per_level import ExpandColumnPerLevel

__all__ = ["HolidayEffect", "FourierHolidayEffect"]


### Probably make FourierHoliday inherit from Holiday
### Change HolidayEffect to BinaryHolidayEffect

class HolidayEffect(BaseAdditiveOrMultiplicativeEffect):
    """Holiday effect for time series forecasting.
    
    This effect models specific holidays or events as additive or multiplicative effects.
    Holidays can have windows before and after the actual date to model effects that
    extend beyond the holiday itself (like pre-Christmas shopping or post-holiday sales).
    Holidays follow a dist.normal(0, prior_scale)
    
    Parameters
    ----------
    holidays : pd.DataFrame
        DataFrame with holidays. Must contain 'holiday' and 'ds' columns.
        Optionally can include:
            - 'lower_window' and 'upper_window' to define custom windows for specific holidays.
            - 'prior_scale' for custom prior_scale values.
    country_name : str, optional
        Country name for built-in holidays. If provided, built-in holidays for this country
        will be added to the user-provided holidays.
    lower_window : int, default 0
        Number of days before the holiday to include in the holiday effect.
        For example, -1 means the holiday effect starts 1 day before the holiday.
    upper_window : int, default 0
        Number of days after the holiday to include in the holiday effect.
        For example, 1 means the holiday effect continues 1 day after the holiday.
    prior_scale : float or dict, default 10.0
        Scale of the prior distribution for the holiday effect. Can be a float applied to
        all holidays or a dictionary mapping holiday names to prior scales.
    effect_mode : str, default 'additive'
        Either "multiplicative" or "additive".
    include_day_of_week : bool, default False
        If True, includes the day of week in holiday identifiers. This allows the model to
        learn different effects for the same holiday depending on which day of the week it
        falls on (e.g., Christmas on Monday vs Christmas on Saturday).
    """

    _tags = {
        "capability:panel": True,
        "requires_X": False,
    }

    def __init__(
        self,
        holidays: Optional[pd.DataFrame] = None,
        country_name: Optional[str] = None,
        lower_window: int = 0,
        upper_window: int = 0,
        prior_scale: float = 0.1,
        include_day_of_week: bool = False,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive"
    ):
        self.holidays = holidays
        self.country_name = country_name
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.prior_scale = prior_scale
        self.include_day_of_week = include_day_of_week

        super().__init__(effect_mode = effect_mode)
        
    def _get_built_in_holidays(self) -> pd.DataFrame:
        """Get built-in holidays for the specified country.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with built-in holidays for the specified country.
        """
        try:
            from holidays import country_holidays  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'holidays' package is required for built-in holidays. "
                "Install it with 'pip install holidays'."
            )
            
        # Get years from the time series data
        years = self._get_years_from_timeindex()
        
        # Create a DataFrame with all holidays for the specified country and years
        country_holiday_dates = []
        
        for year in years:
            holidays_dict = country_holidays(self.country_name, years=year)
            for date, name in holidays_dict.items():
                country_holiday_dates.append({
                    'holiday': name,
                    'ds': date,
                    'country': self.country_name
                })
                
        return pd.DataFrame(country_holiday_dates)
    
    def _get_years_from_timeindex(self) -> List[int]:
        """Extract years from the time index of the training data.
        
        Returns
        -------
        List[int]
            List of unique years in the time index.
        """
        # Get unique dates from the time index
        time_index = self.y_.index.get_level_values(-1)
        
        # Convert to datetime if necessary
        if not isinstance(time_index[0], (datetime, pd.Timestamp)):
            time_index = pd.to_datetime(time_index)
            
        # Extract and return unique years
        return sorted(list(time_index.year.unique()))
    
    def _get_day_of_week_name(self, date: pd.Timestamp) -> str:
        """Get the day of week name for a given date.
        
        Parameters
        ----------
        date : pd.Timestamp
            The date to get the day of week name for.
            
        Returns
        -------
        str
            Day of week name (Monday, Tuesday, etc.)
        """
        return date.day_name()

    def _preprocess_holidays(self) -> pd.DataFrame:
        """Preprocess the holidays DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed holidays DataFrame with expanded holiday windows.
        """
        # Combine user-provided and built-in holidays if applicable
        if self.holidays is None:
            if self.country_name is None:
                raise ValueError("Either 'holidays' or 'country_name' must be provided.")
            holidays_df = self._get_built_in_holidays()
        else:
            # Validate required columns
            if 'holiday' not in self.holidays.columns or 'ds' not in self.holidays.columns:
                raise ValueError("The holidays DataFrame must contain 'holiday' and 'ds' columns.")
                
            if self.country_name is not None:
                # Combine with built-in holidays
                built_in_holidays = self._get_built_in_holidays()
                holidays_df = pd.concat([self.holidays, built_in_holidays], ignore_index=True)
            else:
                holidays_df = self.holidays.copy()
                
        # Convert ds to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(holidays_df['ds']):
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            
        # Add default lower_window and upper_window if not provided
        if 'lower_window' not in holidays_df.columns:
            holidays_df['lower_window'] = self.lower_window
        if 'upper_window' not in holidays_df.columns:
            holidays_df['upper_window'] = self.upper_window
        
        # Add default prior_scale if not provided
        if 'prior_scale' not in holidays_df.columns:
            holidays_df['prior_scale'] = self.prior_scale

        # Expand holidays based on windows
        expanded_holidays = []
        for _, row in holidays_df.iterrows():
            holiday_name = row['holiday']
            holiday_date = row['ds']
            lower_window = row['lower_window']
            upper_window = row['upper_window']
            prior_scale = row['prior_scale']

            if prior_scale is None:
                prior_scale = self.prior_scale
            
            # Create a row for each day in the holiday window
            for offset in range(lower_window, upper_window + 1):
                day = holiday_date + pd.Timedelta(days=offset)
                day_of_week = self._get_day_of_week_name(day)

                # Create the holiday identifier based on whether to include day of week
                if self.include_day_of_week:
                    holiday_id = f"{holiday_name}_{str(offset)}_{day_of_week}"
                else:
                    holiday_id = f"{holiday_name}_{str(offset)}"

                expanded_holidays.append({
                    'holiday': holiday_id,
                    'ds': day.date(),
                    'offset': offset,
                    'day_of_week': day_of_week,
                    'prior_scale': prior_scale,
                    'country': row.get('country', 'user_defined')
                })
                
        return pd.DataFrame(expanded_holidays)
    
    def _create_holiday_features(self, dates_index: pd.Index) -> pd.DataFrame:
        """Create holiday features for the given dates.
        
        Parameters
        ----------
        dates_index : pd.Index
            The dates for which to create holiday features.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with holiday features (one column per holiday-offset combination).
        """
        # Convert dates_index to datetime if needed
        if not isinstance(dates_index[0], (datetime, pd.Timestamp)):
            dates = pd.to_datetime(dates_index)
        else:
            dates = dates_index
            
        # Create a DataFrame with dates as index
        features_df = pd.DataFrame(index=dates)

        # Extract just the date part (without time) for matching
        date_only = pd.Series(dates.date if hasattr(dates, 'date') else 
                             [d.date() for d in dates], index=dates)
        
        # For each holiday in the expanded holidays DataFrame, create a feature column
        for holiday_name in self.holiday_names_:
            # Initialize the feature column with zeros
            col_name = f"holiday_{holiday_name}"
            features_df[col_name] = 0
            
            # Set 1 for dates that match this holiday
            holiday_dates = self.expanded_holidays_[
                self.expanded_holidays_['holiday'] == holiday_name]['ds']
            
            # Match based on date only (ignoring time)
            for holiday_date in holiday_dates:
                # Find all timestamps that fall on this date
                matching_indices = dates[date_only == holiday_date]
                
                if len(matching_indices) > 0:
                    features_df.loc[matching_indices, col_name] = 1
                
        return features_df
    
    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Initialize the holiday effect.
        
        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe.
        X : pd.DataFrame
            The DataFrame to initialize the effect.
        scale : float, optional
            The scale of the timeseries, by default 1.0.
        """
        self.y_ = y
        
        # Preprocess holidays
        self.expanded_holidays_ = self._preprocess_holidays()
        
        # Get unique holiday names
        self.holiday_names_ = self.expanded_holidays_['holiday'].unique()
        
        # Store the prior scales for each holiday
        self.holiday_prior_scales_ = self.expanded_holidays_[['holiday', 'prior_scale']].set_index("holiday").to_dict()['prior_scale']
            
        # Store the scale for later use
        self.scale_ = scale
        
        # Store information about the time index for later use
        self.index_names_ = y.index.names
        self.has_multiple_series_ = y.index.nlevels > 1
        if self.has_multiple_series_:
            self.series_idx_ = y.index.droplevel(-1).unique()
    
    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        """Transform input data to prepare for prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        fh : pd.Index
            The forecasting horizon as a pandas Index.
            
        Returns
        -------
        Dict
            Dictionary with holiday features and metadata.
        """
        # Get the time index
        if self.has_multiple_series_:
            # For multivariate time series, we need to reconstruct the full index
            idx_list = self.series_idx_.to_list()
            idx_list = [x if isinstance(x, tuple) else (x,) for x in idx_list]
            
            # Create a new multi-index combining series and forecast horizon
            idx_tuples = []
            for series_idx in idx_list:
                for time_idx in fh:
                    idx_tuples.append((*series_idx, time_idx))
                    
            idx = pd.MultiIndex.from_tuples(idx_tuples, names=self.index_names_)
            time_idx = fh
        else:
            # For univariate time series, the index is just the forecast horizon
            idx = fh
            time_idx = fh
            
        # Create holiday features
        holiday_features = self._create_holiday_features(time_idx) # rows: dates // cols: unique_holiday
        
        # Convert to array
        n_holidays = len(self.holiday_names_)
        if self.has_multiple_series_:
            n_series = len(self.series_idx_)
            n_times = len(fh)
            
            # Reshape features for multivariate case
            features_array = np.zeros((n_series, n_times, n_holidays))
            
            for i, series_idx in enumerate(idx_list):
                for j, time_idx in enumerate(fh):
                    for k, holiday in enumerate(self.holiday_names_):
                        col = f"holiday_{holiday}"
                        if col in holiday_features.columns:
                            features_array[i, j, k] = holiday_features.loc[time_idx, col]
        else:
            # For univariate case, simpler reshaping
            features_array = np.zeros((len(fh), n_holidays))
            for i, time_idx in enumerate(fh):
                for j, holiday in enumerate(self.holiday_names_):
                    col = f"holiday_{holiday}"
                    if col in holiday_features.columns:
                        features_array[i, j] = holiday_features.loc[time_idx, col]
                        
        # Return as dictionary with features and metadata
        return {
            'features': jnp.array(features_array),
            'holiday_names': self.holiday_names_,
            'prior_scales': self.holiday_prior_scales_,
            'has_multiple_series': self.has_multiple_series_,
            'effect_mode': self.effect_mode
        }
    
    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the holiday effect.
        
        Parameters
        ----------
        data : Dict
            Dictionary with holiday features and metadata.
        predicted_effects : Dict[str, jnp.ndarray]
            Dictionary with other predicted effects.
            
        Returns
        -------
        jnp.ndarray
            The holiday effect values.
        """
        features = data['features']
        holiday_names = data['holiday_names']
        prior_scales = data['prior_scales']
        has_multiple_series = data['has_multiple_series']
        effect_mode = data['effect_mode']
        
        # Sample holiday coefficients
        holiday_coefs = {}
        for holiday in holiday_names:
            prior_scale = prior_scales[holiday]
            
            # For each holiday, sample a coefficient from a normal distribution
            holiday_coefs[holiday] = numpyro.sample(
                f"holiday_{holiday}",
                dist.Normal(0, prior_scale)
            )
            
        # Combine coefficients into a single vector
        beta = jnp.array([holiday_coefs[holiday] for holiday in holiday_names])
        
        # Apply the holiday effect
        if has_multiple_series:
            # For multivariate case, we need to broadcast properly
            # Reshape beta to (1, 1, n_holidays) for broadcasting
            beta_reshaped = beta.reshape(1, 1, -1)
            holiday_effect = jnp.sum(features * beta_reshaped, axis=-1, keepdims=True)
        else:
            # For univariate case
            holiday_effect = jnp.sum(features * beta, axis=-1, keepdims=True)
            
        return holiday_effect
        
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get test parameters for the holiday effect.
        
        Parameters
        ----------
        parameter_set : str, default "default"
            Parameter set to use.
            
        Returns
        -------
        List
            List of parameter sets.
        """
        # Create a simple holiday DataFrame for testing
        holidays_df = pd.DataFrame({
            'holiday': ['New Year', 'Christmas'],
            'ds': [f'2023-01-01', f'2023-12-25'],
        })
        
        return [
            {
                'holidays': holidays_df,
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 10.0,
                'effect_mode': 'additive',
            }
        ]

class FourierHolidayEffect(BaseAdditiveOrMultiplicativeEffect):
    """Fourier-based holiday effect for granular time series forecasting.
    
    This effect models holidays for sub-daily (e.g., hourly) data using Fourier series.
    Instead of having a separate parameter for each hour of a holiday, it uses
    Fourier coefficients to represent smooth intraday patterns for each holiday.
    This approach is more parameter-efficient and can prevent overfitting.
    Solves the following issues of Binary Indicators:
        - Parameter Explosion: For each holiday and each hour, you need a separate parameter, which can lead to overfitting
        - Discontinuous Effects: Hour-by-hour effects can be jumpy and unrealistic
        - Poor Generalization: The model cannot easily transfer knowledge between similar hours    

    Parameters
    ----------
    holidays : pd.DataFrame
        DataFrame with holidays. Must contain 'holiday' and 'ds' columns.
        The 'ds' column should contain the date (not datetime) of the holiday.
        Optionally can include:
            - 'lower_window' and 'upper_window' to define custom windows for specific holidays.
            - 'prior_scale' for custom prior_scale values.
    country_name : str, optional
        Country name for built-in holidays. If provided, built-in holidays for this country
        will be added to the user-provided holidays.
    lower_window : int, default 0
        Number of days before the holiday to include in the holiday effect.
        For example, -1 means the holiday effect starts 1 day before the holiday.
    upper_window : int, default 0
        Number of days after the holiday to include in the holiday effect.
        For example, 1 means the holiday effect continues 1 day after the holiday.
    fourier_terms : int, default 3
        Number of Fourier terms to use for modeling intraday patterns.
        Higher values allow for more complex patterns but may lead to overfitting.
    granularity : str, default 'h'
        Frequency of the time series. Example: "H" for hourly, "T" or "min" for minutes.
    periods_per_day : int, optional
        Number of periods per day (e.g., 24 for hourly data). If not provided,
        it will be inferred from the granularity.
    prior_scale : float, default 0.1
        Scale of the prior distribution for the holiday effect coefficients.
    include_day_of_week : bool, default False
        If True, creates separate parameters for holidays falling on different days of the week.
    effect_mode : str, default 'additive'
        Either "multiplicative" or "additive".
    """

    _tags = {
        "capability:panel": True,
        "requires_X": False,
    }

    def __init__(
        self,
        holidays: Optional[pd.DataFrame] = None,
        country_name: Optional[str] = None,
        lower_window: int = 0,
        upper_window: int = 0,
        fourier_terms: int = 3,
        granularity: str = 'h',
        periods_per_day: Optional[int] = None,
        prior_scale: float = 0.1,
        include_day_of_week: bool = False,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive"
    ):
        self.holidays = holidays
        self.country_name = country_name
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.fourier_terms = fourier_terms
        self.granularity = granularity
        self.prior_scale = prior_scale
        self.include_day_of_week = include_day_of_week
        
        # Determine periods per day based on granularity if not provided
        if periods_per_day is None:
            if granularity == 'h':
                self.periods_per_day = 24
            elif granularity in ['T', 'min']:
                self.periods_per_day = 24 * 60
            elif granularity == '30T':
                self.periods_per_day = 48
            elif granularity == '15T':
                self.periods_per_day = 96
            else:
                raise ValueError(f"Cannot infer periods_per_day for granularity {granularity}. "
                                "Please provide periods_per_day explicitly.")
        else:
            self.periods_per_day = periods_per_day

        self.fourier_features_ = None
        self.holiday_linear_effects_ = {}
        self.expand_column_per_level_ = None

        super().__init__(effect_mode=effect_mode)
    
    def _get_built_in_holidays(self) -> pd.DataFrame:
        """Get built-in holidays for the specified country.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with built-in holidays for the specified country.
        """
        try:
            from holidays import country_holidays  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'holidays' package is required for built-in holidays. "
                "Install it with 'pip install holidays'."
            )
            
        # Get years from the time series data
        years = self._get_years_from_timeindex()
        
        # Create a DataFrame with all holidays for the specified country and years
        country_holiday_dates = []
        
        for year in years:
            holidays_dict = country_holidays(self.country_name, years=year)
            for date, name in holidays_dict.items():
                country_holiday_dates.append({
                    'holiday': name,
                    'ds': date,
                    'country': self.country_name
                })
                
        return pd.DataFrame(country_holiday_dates)
    
    def _get_years_from_timeindex(self) -> List[int]:
        """Extract years from the time index of the training data.
        
        Returns
        -------
        List[int]
            List of unique years in the time index.
        """
        # Get unique dates from the time index
        time_index = self.y_.index.get_level_values(-1)
        
        # Convert to datetime if necessary
        if not isinstance(time_index[0], (datetime, pd.Timestamp)):
            time_index = pd.to_datetime(time_index)
            
        # Extract and return unique years
        return sorted(list(time_index.year.unique()))
    
    def _get_day_of_week_name(self, date: pd.Timestamp) -> str:
        """Get the day of week name for a given date.
        
        Parameters
        ----------
        date : pd.Timestamp
            The date to get the day of week name for.
            
        Returns
        -------
        str
            Day of week name (Monday, Tuesday, etc.)
        """
        return date.day_name()
    
    def _preprocess_holidays(self) -> pd.DataFrame:
        """Preprocess the holidays DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed holidays DataFrame with expanded holiday windows.
        """
        # Combine user-provided and built-in holidays if applicable
        if self.holidays is None:
            if self.country_name is None:
                raise ValueError("Either 'holidays' or 'country_name' must be provided.")
            holidays_df = self._get_built_in_holidays()
        else:
            # Validate required columns
            if 'holiday' not in self.holidays.columns or 'ds' not in self.holidays.columns:
                raise ValueError("The holidays DataFrame must contain 'holiday' and 'ds' columns.")
                
            if self.country_name is not None:
                # Combine with built-in holidays
                built_in_holidays = self._get_built_in_holidays()
                holidays_df = pd.concat([self.holidays, built_in_holidays], ignore_index=True)
            else:
                holidays_df = self.holidays.copy()
                
        # Convert ds to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(holidays_df['ds']):
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            
        # Add default lower_window and upper_window if not provided
        if 'lower_window' not in holidays_df.columns:
            holidays_df['lower_window'] = self.lower_window
        if 'upper_window' not in holidays_df.columns:
            holidays_df['upper_window'] = self.upper_window
        
        # Add default prior_scale if not provided
        if 'prior_scale' not in holidays_df.columns:
            holidays_df['prior_scale'] = self.prior_scale

        # Expand holidays based on windows
        expanded_holidays = []
        for _, row in holidays_df.iterrows():
            holiday_name = row['holiday']
            holiday_date = row['ds']
            lower_window = row['lower_window']
            upper_window = row['upper_window']
            prior_scale = row['prior_scale']

            if prior_scale is None:
                prior_scale = self.prior_scale
            
            # Create a row for each day in the holiday window
            for offset in range(lower_window, upper_window + 1):
                day = holiday_date + pd.Timedelta(days=offset)
                day_of_week = self._get_day_of_week_name(day)
                
                # Create the holiday identifier based on whether to include day of week
                if self.include_day_of_week:
                    holiday_id = f"{holiday_name}_{str(offset)}_{day_of_week}"
                else:
                    holiday_id = f"{holiday_name}_{str(offset)}"

                expanded_holidays.append({
                    'holiday': holiday_id,
                    'ds': day.date(),  # Store just the date part
                    'offset': offset,
                    'day_of_week': day_of_week,
                    'prior_scale': prior_scale,
                    'country': row.get('country', 'user_defined')
                })
                
        return pd.DataFrame(expanded_holidays)
    
    def _create_holiday_indicators(self, dates_index: pd.Index) -> pd.DataFrame:
        """Create holiday indicator features for the given dates.
        
        Parameters
        ----------
        dates_index : pd.Index
            The dates for which to create holiday features.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with holiday indicator features (one column per holiday-offset combination).
        """
        # Convert dates_index to datetime if needed
        if not isinstance(dates_index[0], (datetime, pd.Timestamp)):
            dates = pd.to_datetime(dates_index)
        else:
            dates = dates_index.copy()
        
        # Create a DataFrame with dates as index
        features_df = pd.DataFrame(index=dates)
        
        # Extract just the date part (without time) for matching
        date_only = pd.Series(dates.date if hasattr(dates, 'date') else 
                             [d.date() for d in dates], index=dates)
        
        # For each holiday in the expanded holidays DataFrame, create a feature column
        for holiday_name in self.holiday_names_:
            # Initialize the feature column with zeros
            col_name = f"holiday_{holiday_name}"
            features_df[col_name] = 0
            
            # Get the dates for this holiday
            holiday_dates = self.expanded_holidays_[
                self.expanded_holidays_['holiday'] == holiday_name]['ds']
            
            # Match based on date only (ignoring time)
            for holiday_date in holiday_dates:
                # Find all timestamps that fall on this date
                matching_indices = dates[date_only == holiday_date]
                
                if len(matching_indices) > 0:
                    features_df.loc[matching_indices, col_name] = 1
                
        return features_df
    
    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Initialize the holiday effect.
        
        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe.
        X : pd.DataFrame
            The DataFrame to initialize the effect.
        scale : float, optional
            The scale of the timeseries, by default 1.0.
        """
        self.y_ = y
        
        # Preprocess holidays
        self.expanded_holidays_ = self._preprocess_holidays()
        
        # Get unique holiday names
        self.holiday_names_ = self.expanded_holidays_['holiday'].unique()
        
        # Store the prior scales for each holiday
        self.holiday_prior_scales_ = self.expanded_holidays_[['holiday', 'prior_scale']] \
                                    .set_index("holiday").to_dict()['prior_scale']
            
        # Initialize the Fourier features transformer
        self.fourier_features_ = FourierFeatures(
            sp_list=[self.periods_per_day],
            fourier_terms_list=[self.fourier_terms],
            freq=self.granularity,
            keep_original_columns=False
        )
        
        # Fit the Fourier features transformer on the time index
        date_df = pd.DataFrame(index=X.index)
        self.fourier_features_.fit(X=date_df)
        
        # Create holiday indicators for the training period
        holiday_indicators = self._create_holiday_indicators(X.index)
        
        # Create LinearEffect for each holiday
        self.holiday_linear_effects_ = {}
        
        for holiday_name in self.holiday_names_:
            col_name = f"holiday_{holiday_name}"
            
            # Skip if this holiday doesn't appear in the training data
            if col_name not in holiday_indicators.columns or not holiday_indicators[col_name].any():
                continue
            
            # Create a copy of X with just the rows for this holiday
            holiday_mask = holiday_indicators[col_name] == 1
            if not holiday_mask.any():
                continue
                
            # Get Fourier features for the entire dataset
            fourier_features = self.fourier_features_.transform(date_df)
            
            # Multiply by holiday indicator to zero out non-holiday times
            holiday_fourier = fourier_features.copy()
            for col in holiday_fourier.columns:
                holiday_fourier[col] = holiday_fourier[col] * holiday_indicators[col_name]
            
            # Create a LinearEffect for this holiday
            prior_scale = self.holiday_prior_scales_[holiday_name]
            holiday_effect = LinearEffect(
                prior=dist.Normal(0, prior_scale),
                effect_mode=self.effect_mode
            )
            
            # Fit the LinearEffect
            holiday_effect.fit(X=holiday_fourier, y=y, scale=scale)
            
            # Store the fitted effect
            self.holiday_linear_effects_[holiday_name] = holiday_effect
        
        # Store info about multivariate series
        self.index_names_ = y.index.names
        self.has_multiple_series_ = y.index.nlevels > 1
        if self.has_multiple_series_:
            self.series_idx_ = y.index.droplevel(-1).unique()
            
        # Initialize ExpandColumnPerLevel if needed
        X_fourier = self.fourier_features_.transform(date_df)
        if X_fourier.index.nlevels > 1 and X_fourier.index.droplevel(-1).nunique() > 1:
            self.expand_column_per_level_ = ExpandColumnPerLevel([".*"]).fit(X=X_fourier)
            
        # Store the scale for later use
        self.scale_ = scale
    
    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        """Transform input data to prepare for prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        fh : pd.Index
            The forecasting horizon as a pandas Index.
            
        Returns
        -------
        Dict
            Dictionary with holiday data for prediction.
        """
        # Get the time index
        if self.has_multiple_series_:
            # For multivariate time series, we need to reconstruct the full index
            idx_list = self.series_idx_.to_list()
            idx_list = [x if isinstance(x, tuple) else (x,) for x in idx_list]
            
            # Create a new multi-index combining series and forecast horizon
            idx_tuples = []
            for series_idx in idx_list:
                for time_idx in fh:
                    idx_tuples.append((*series_idx, time_idx))
                    
            idx = pd.MultiIndex.from_tuples(idx_tuples, names=self.index_names_)
            time_idx = fh
        else:
            # For univariate time series, the index is just the forecast horizon
            idx = fh
            time_idx = fh
            
        # Create a DataFrame with the time index
        date_df = pd.DataFrame(index=idx)
        
        # Create holiday indicators
        holiday_indicators = self._create_holiday_indicators(time_idx)
        
        # Get Fourier features for the forecast horizon
        fourier_features = self.fourier_features_.transform(date_df)
        
        # Apply ExpandColumnPerLevel if needed
        if self.expand_column_per_level_ is not None:
            fourier_features = self.expand_column_per_level_.transform(fourier_features)
        
        # Create holiday-specific Fourier features
        holiday_data = {}
        
        for holiday_name in self.holiday_names_:
            col_name = f"holiday_{holiday_name}"
            
            # Skip if this holiday doesn't appear in the forecast period
            if col_name not in holiday_indicators.columns or not holiday_indicators[col_name].any():
                continue
                
            # Skip if no LinearEffect was created for this holiday
            if holiday_name not in self.holiday_linear_effects_:
                continue
                
            # Multiply by holiday indicator to zero out non-holiday times
            holiday_fourier = fourier_features.copy()
            for col in holiday_fourier.columns:
                holiday_fourier[col] = holiday_fourier[col] * holiday_indicators[col_name]
            
            # Get transformed data for this holiday
            holiday_effect = self.holiday_linear_effects_[holiday_name]
            holiday_data[holiday_name] = holiday_effect.transform(holiday_fourier, fh)
        
        return {
            'holiday_data': holiday_data,
            'effect_mode': self.effect_mode
        }
    
    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the holiday effect.
        
        Parameters
        ----------
        data : Dict
            Dictionary with holiday data for prediction.
        predicted_effects : Dict[str, jnp.ndarray]
            Dictionary with other predicted effects.
            
        Returns
        -------
        jnp.ndarray
            The holiday effect values.
        """
        holiday_data = data['holiday_data']
        
        # Sum up effects from all holidays
        for holiday_name, holiday_features in holiday_data.items():
            # Get the LinearEffect for this holiday
            holiday_effect_model = self.holiday_linear_effects_[holiday_name]
            
            # Predict the effect for this holiday
            holiday_effect = holiday_effect_model.predict(holiday_features, predicted_effects)
        
        return holiday_effect
        
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get test parameters for the holiday effect.
        
        Parameters
        ----------
        parameter_set : str, default "default"
            Parameter set to use.
            
        Returns
        -------
        List
            List of parameter sets.
        """
        # Create a simple holiday DataFrame for testing
        holidays_df = pd.DataFrame({
            'holiday': ['New Year', 'Christmas'],
            'ds': [f'2023-01-01', f'2023-12-25'],
        })
        
        return [
            {
                'holidays': holidays_df,
                'lower_window': -1,
                'upper_window': 1,
                'fourier_terms': 3,
                'granularity': 'h',
                'prior_scale': 0.1,
                'effect_mode': 'additive',
            }
        ]