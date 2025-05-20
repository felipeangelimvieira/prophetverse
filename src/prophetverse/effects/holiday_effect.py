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

__all__ = ["HolidayEffect"]


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

                # Create the holiday identifier based on whether to include day of week
                if self.include_day_of_week:
                    day_of_week = self._get_day_of_week_name(day)
                    holiday_id = f"{holiday_name}_{str(offset)}_{day_of_week}"
                else:
                    holiday_id = f"{holiday_name}_{str(offset)}"

                expanded_holidays.append({
                    'holiday': holiday_id,
                    'ds': day,
                    'offset': offset,
                    'day_of_week': self._get_day_of_week_name(day),
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
        
        # For each holiday in the expanded holidays DataFrame, create a feature column
        for holiday_name in self.holiday_names_:
            # Initialize the feature column with zeros
            col_name = f"holiday_{holiday_name}"
            features_df[col_name] = 0
            
            # Set 1 for dates that match this holiday
            holiday_dates = self.expanded_holidays_[
                self.expanded_holidays_['holiday'] == holiday_name]['ds']
            
            # Find matching dates
            matching_dates = dates[dates.isin(holiday_dates)]
            if len(matching_dates) > 0:
                features_df.loc[matching_dates, col_name] = 1
                
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