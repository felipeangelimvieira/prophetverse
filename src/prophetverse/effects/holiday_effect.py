"""Holiday effects for time series forecasting."""

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from datetime import datetime, timedelta

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseAdditiveOrMultiplicativeEffect, BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor
from prophetverse.effects.linear import LinearEffect
from sktime.transformations.series.fourier import FourierFeatures
from prophetverse.sktime._expand_column_per_level import ExpandColumnPerLevel

__all__ = ["BinaryHolidayEffect", "FourierHolidayEffect", "BinaryStdHolidayEffect"]


# Binary is not using LinearEffect because prior_scales can differ by holiday.
# LinearEffect vectorizes the calculation to 1 prior_scale.
# That's why it's slower.
class BinaryHolidayEffect(BaseAdditiveOrMultiplicativeEffect):
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
    prior_scale : float or dict, default 0.1
        Scale of the prior distribution for the holiday effect. Can be a float applied to
        all holidays or a dictionary mapping holiday names to prior scales.
    effect_mode : str, default 'additive'
        Either "multiplicative" or "additive".
    include_day_of_week : bool, default False
        If True, includes the day of week in holiday identifiers. This allows the model to
        learn different effects for the same holiday depending on which day of the week it
        falls on (e.g., Christmas on Monday vs Christmas on Saturday).
    daily_splits : int, default 1
        Number of batches to split each day into for granular time series data.
        For example, with hourly data (24 periods/day), daily_splits=2 creates 
        12-hour blocks, daily_splits=4 creates 6-hour blocks. Must be a divisor of
        periods_per_day. If not a valid divisor, the closest valid divisor ≤ daily_splits
        will be used.
    granularity : str, default 'h'
        Frequency of the time series. Used to infer periods_per_day if not provided.
        Examples: 'h' for hourly, 'T' for minutes, '30T' for 30-minute intervals.
    periods_per_day : int, optional
        Number of periods per day (e.g., 24 for hourly data). If not provided,
        it will be inferred from the granularity.
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
        daily_splits: int = 1,
        granularity: str = 'h',
        periods_per_day: Optional[int] = None,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive"
    ):
        self.holidays = holidays
        self.country_name = country_name
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.prior_scale = prior_scale
        self.include_day_of_week = include_day_of_week
        self.daily_splits = daily_splits
        self.granularity = granularity
        super().__init__(effect_mode = effect_mode)
        
        # Determine periods per day based on granularity if not provided
        if periods_per_day is None:
            if granularity.lower() in ['h', 'hour']:
                self.periods_per_day = 24
            elif granularity in ['T', 'min']:
                self.periods_per_day = 24 * 60
            elif granularity == '30T':
                self.periods_per_day = 24 * 2
            elif granularity == '15T':
                self.periods_per_day = 24 * 4
            else:
                # Default to 24 (hourly) if cannot infer
                self.periods_per_day = 24
        else:
            self.periods_per_day = periods_per_day
            
        # Validate and adjust daily_splits
        self.daily_splits = self._validate_daily_splits(daily_splits, self.periods_per_day)
        
        # Attributes to be populated by _fit
        self.expanded_holidays_: Optional[pd.DataFrame] = None
        self.holiday_names_: Optional[List[str]] = None
        self.holiday_prior_scales_: Optional[Dict[str, float]] = None
        self.y_ = None # To store y for year inference if needed by country holidays
        self.index_names_ = None
        self.has_multiple_series_ = False
        self.series_idx_ = None
        self.scale_ = None

    def _validate_daily_splits(self, daily_splits: int, periods_per_day: int) -> int:
        """Validate and adjust daily_splits to be a valid divisor of periods_per_day.
        
        Parameters
        ----------
        daily_splits : int
            The requested number of daily splits.
        periods_per_day : int
            The number of periods per day.
            
        Returns
        -------
        int
            The validated daily_splits value.
        """
        if daily_splits <= 0:
            raise ValueError("daily_splits must be positive")
            
        if daily_splits == 1:
            return daily_splits
            
        # Find all valid divisors of periods_per_day
        valid_divisors = []
        for i in range(1, periods_per_day + 1):
            if periods_per_day % i == 0:
                valid_divisors.append(i)
        
        # If daily_splits is already valid, use it
        if daily_splits in valid_divisors:
            return daily_splits
            
        # Find the closest valid divisor that is <= daily_splits
        valid_divisors_le = [d for d in valid_divisors if d <= daily_splits]
        if valid_divisors_le:
            closest = max(valid_divisors_le)
            print(f"Warning: daily_splits={daily_splits} is not a valid divisor of periods_per_day={periods_per_day}. "
                    f"Using closest valid divisor: {closest}")
            return closest
        else:
            # If no valid divisor <= daily_splits, use 1
            print(f"Warning: daily_splits={daily_splits} is not valid for periods_per_day={periods_per_day}. "
                    f"Using daily_splits=1")
            return 1

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
        self.y_ = y # Store y primarily for _get_holiday_years if country_name is used
        self.scale_ = scale
        self.index_names_ = y.index.names
        self.has_multiple_series_ = y.index.nlevels > 1
        if self.has_multiple_series_:
            self.series_idx_ = y.index.droplevel(-1).unique()

        time_idx_for_years = self.y_.index.get_level_values(-1) if self.y_ is not None else None

        expanded_holidays_df = _preprocess_holiday_df(
            raw_holidays_df=self.holidays,
            country_name=self.country_name,
            time_index_for_years=time_idx_for_years,
            default_lower_window=self.lower_window,
            default_upper_window=self.upper_window,
            default_prior_scale=self.prior_scale,
            include_day_of_week=self.include_day_of_week
        )
        self.expanded_holidays_ = expanded_holidays_df
        
        if not self.expanded_holidays_.empty:
            self.holiday_names_ = self.expanded_holidays_['holiday'].unique().tolist()
            self.holiday_prior_scales_ = self.expanded_holidays_[
                ['holiday', 'prior_scale']
            ].drop_duplicates().set_index("holiday")['prior_scale'].to_dict()
        else:
            self.holiday_names_ = []
            self.holiday_prior_scales_ = {}
            
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
            time_idx = fh # For _create_holiday_features_df, we need a unique time index.
            idx_list = []  # Initialize for type checker
            
        # Create holiday features using the helper function
        holiday_features_df = _create_batched_holiday_features_df(
            dates_index=time_idx, # This should be the unique time points in fh
            expanded_holidays_df=self.expanded_holidays_,
            holiday_names=self.holiday_names_,
            daily_splits=self.daily_splits,
            periods_per_day=self.periods_per_day
        )
        
        # Convert to array (logic updated for batching support)
        if self.daily_splits == 1:
            # Original logic for no batching
            n_holidays = len(self.holiday_names_)
            if self.has_multiple_series_:
                n_series = len(self.series_idx_)
                n_times = len(fh)
                
                # Reshape features for multivariate case
                features_array = np.zeros((n_series, n_times, n_holidays))
                
                for i, series_idx in enumerate(idx_list):
                    for j, time_idx in enumerate(fh):
                        for k, holiday in enumerate(self.holiday_names_):
                            col_name = f"holiday_{holiday}"
                            if col_name in holiday_features_df.columns:
                                features_array[i, j, k] = holiday_features_df.loc[fh[j], col_name]
            else:
                # For univariate case, simpler reshaping
                features_array = np.zeros((len(fh), n_holidays))
                for i, time_idx_val in enumerate(fh):
                    for j, holiday_name in enumerate(self.holiday_names_):
                        col_name = f"holiday_{holiday_name}"
                        if col_name in holiday_features_df.columns:
                            features_array[i, j] = holiday_features_df.loc[time_idx_val, col_name]
        else:
            # Batching logic - each holiday gets multiple columns (one per batch)
            n_holidays_batched = len(self.holiday_names_) * self.daily_splits
            if self.has_multiple_series_:
                n_series = len(self.series_idx_)
                n_times = len(fh)
                
                # Reshape features for multivariate case with batching
                features_array = np.zeros((n_series, n_times, n_holidays_batched))
                
                for i, series_idx in enumerate(idx_list):
                    for j, time_idx in enumerate(fh):
                        feature_idx = 0
                        for holiday in self.holiday_names_:
                            for batch_idx in range(self.daily_splits):
                                col_name = f"holiday_{holiday}_batch_{batch_idx}"
                                if col_name in holiday_features_df.columns:
                                    features_array[i, j, feature_idx] = holiday_features_df.loc[fh[j], col_name]
                                feature_idx += 1
            else:
                # For univariate case with batching
                features_array = np.zeros((len(fh), n_holidays_batched))
                for i, time_idx_val in enumerate(fh):
                    feature_idx = 0
                    for holiday_name in self.holiday_names_:
                        for batch_idx in range(self.daily_splits):
                            col_name = f"holiday_{holiday_name}_batch_{batch_idx}"
                            if col_name in holiday_features_df.columns:
                                features_array[i, feature_idx] = holiday_features_df.loc[time_idx_val, col_name]
                            feature_idx += 1
                        
        # Return as dictionary with features and metadata
        return {
            'features': jnp.array(features_array),
            'holiday_names': self.holiday_names_,
            'prior_scales': self.holiday_prior_scales_,
            'has_multiple_series': self.has_multiple_series_,
            'effect_mode': self.effect_mode,
            'daily_splits': self.daily_splits
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
        daily_splits = data['daily_splits']
        
        # Sample holiday coefficients
        holiday_coefs = {}
        
        if daily_splits == 1:
            # Original logic: one parameter per holiday
            for holiday in holiday_names:
                prior_scale = prior_scales[holiday]
                
                # For each holiday, sample a coefficient from a normal distribution
                holiday_coefs[holiday] = numpyro.sample(
                    f"holiday_{holiday}",
                    dist.Normal(0, prior_scale)
                )
                
            # Combine coefficients into a single vector
            beta = jnp.array([holiday_coefs[holiday] for holiday in holiday_names])
        else:
            # Batching logic: one parameter per holiday-batch combination
            beta_list = []
            for holiday in holiday_names:
                prior_scale = prior_scales[holiday]
                for batch_idx in range(daily_splits):
                    param_name = f"holiday_{holiday}_batch_{batch_idx}"
                    holiday_coefs[param_name] = numpyro.sample(
                        param_name,
                        dist.Normal(0, prior_scale)
                    )
                    beta_list.append(holiday_coefs[param_name])
            
            # Combine coefficients into a single vector
            beta = jnp.array(beta_list)
        
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
    def get_test_params(cls, parameter_set="default") -> List[Dict[str, Any]]:  # type: ignore[override]
        """Get test parameters for the holiday effect.
        
        Parameters
        ----------
        parameter_set : str, default "default"
            Parameter set to use.
            
        Returns
        -------
        List[Dict[str, Any]]
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
            },
            {
                'holidays': holidays_df,
                'lower_window': 0,
                'upper_window': 0,
                'prior_scale': 5.0,
                'effect_mode': 'additive',
                'daily_splits': 2,
                'granularity': 'h',
            }
        ]

class FourierHolidayEffect(BaseEffect):
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
        self.prior_scale = prior_scale
        self.include_day_of_week = include_day_of_week
        self.fourier_terms = fourier_terms
        self.granularity = granularity
        # super().__init__(effect_mode=effect_mode)
        self.effect_mode = effect_mode
        super().__init__()

        
        # Determine periods per day based on granularity if not provided
        if periods_per_day is None:
            if granularity.lower() in ['h', 'hour']:
                self.periods_per_day = 24
            elif granularity in ['T', 'min']:
                self.periods_per_day = 24 * 60
            elif granularity == '30T':
                self.periods_per_day = 24 * 2
            elif granularity == '15T':
                self.periods_per_day = 24 * 4
            else:
                raise ValueError(f"Cannot infer periods_per_day for granularity {granularity}. "
                                "Please provide periods_per_day explicitly.")
        else:
            self.periods_per_day = periods_per_day

        self.fourier_features_ = None
        self.holiday_linear_effects_ = {}
        self.expand_column_per_level_ = None

        self.linear_effect_ = LinearEffect(
            effect_mode=self.effect_mode,
            prior=dist.Normal(0, self.prior_scale)
        )

    
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
        # Get time index for determining years
        time_idx = y.index.get_level_values(-1) if y.index.nlevels > 1 else y.index
        time_idx_for_years = time_idx
        
        # Preprocess holidays
        expanded_holidays_df = _preprocess_holiday_df(
            raw_holidays_df=self.holidays,
            country_name=self.country_name,
            time_index_for_years=time_idx_for_years,
            default_lower_window=self.lower_window,
            default_upper_window=self.upper_window,
            default_prior_scale=self.prior_scale,
            include_day_of_week=self.include_day_of_week
        )
        self.expanded_holidays_ = expanded_holidays_df
        
        if not self.expanded_holidays_.empty:
            self.holiday_names_ = list(self.expanded_holidays_['holiday'].unique())
            # Store prior scales for each holiday
            self.holiday_prior_scales_ = {}
            for holiday in self.holiday_names_:
                holiday_rows = self.expanded_holidays_[self.expanded_holidays_['holiday'] == holiday]
                self.holiday_prior_scales_[holiday] = holiday_rows['prior_scale'].iloc[0]
        else:
            self.holiday_names_ = []
            self.holiday_prior_scales_ = {}
            
        # Store the scale for later use
        self.scale_ = scale
        
        # Store information about the time index for later use
        self.index_names_ = y.index.names
        self.has_multiple_series_ = y.index.nlevels > 1
        if self.has_multiple_series_:
            self.series_idx_ = y.index.droplevel(-1).unique()
            
        # Initialize the Fourier features transformer
        if self.fourier_terms > 0:
            self.fourier_features_ = FourierFeatures(
                sp_list=[self.periods_per_day],
                fourier_terms_list=[self.fourier_terms],
                freq=self.granularity,
                keep_original_columns=False
            )
            
            # Fit the Fourier features transformer on the time index
            date_df = pd.DataFrame(index=X.index)
            self.fourier_features_.fit(X=date_df)
        else:
            # When fourier_terms=0, don't create FourierFeatures to avoid division by zero
            self.fourier_features_ = None
            date_df = pd.DataFrame(index=X.index)
        
        # Store info about multivariate series
        self.index_names_ = y.index.names
        self.has_multiple_series_ = y.index.nlevels > 1
        if self.has_multiple_series_:
            self.series_idx_ = y.index.droplevel(-1).unique()
            
        # Initialize ExpandColumnPerLevel if needed
        # This should operate on the fourier features generated by self.fourier_features_
        # The X used here should be the fourier features, not the original X
        if self.fourier_features_ is not None:
            X_fourier_transformed = self.fourier_features_.transform(date_df)
            # Type assertion to help pylance understand this is a DataFrame
            X_fourier_df: pd.DataFrame = X_fourier_transformed  # type: ignore
            if self.has_multiple_series_ and X_fourier_df.index.droplevel(-1).nunique() > 1:
                self.expand_column_per_level_ = ExpandColumnPerLevel([".*"]).fit(X=X_fourier_df)
            
        # Prepare X_transformed for the linear effect
        # We need to create the same combined features (holiday indicators * fourier features)
        # that will be used during transform/predict
        
        if self.fourier_terms == 0:
            # When fourier_terms is 0, fit LinearEffect with a dummy zero column
            # to avoid empty DataFrame issues in conversion utilities
            empty_X_for_linear = pd.DataFrame(index=date_df.index)
            empty_X_for_linear['dummy_zero'] = 0.0
            self.linear_effect_.fit(X=empty_X_for_linear, y=y, scale=scale)
        else:
            # Create combined holiday-fourier features for training
            time_idx = date_df.index.get_level_values(-1) if self.has_multiple_series_ else date_df.index
            
            # Get holiday indicators for training data
            holiday_indicators_df = _create_holiday_features_df(
                dates_index=time_idx,
                expanded_holidays_df=self.expanded_holidays_,
                holiday_names=self.holiday_names_
            )
            
            # Create combined features: holiday indicators * fourier features
            combined_features_list = []
            
            for holiday_name in self.holiday_names_:
                col_name = f"holiday_{holiday_name}"
                if col_name in holiday_indicators_df.columns:
                    holiday_indicator = holiday_indicators_df[col_name]
                    
                    # Create expanded indicator for the full training index
                    expanded_indicator_df = pd.DataFrame(index=date_df.index)
                    expanded_indicator_df[f'{col_name}_indicator'] = 0.0
                    
                    # Fill in the indicator values
                    if self.has_multiple_series_:
                        series_list = self.series_idx_.to_list()
                        series_list = [x if isinstance(x, tuple) else (x,) for x in series_list]
                        for series_idx in series_list:
                            for time_idx_val in time_idx:
                                if time_idx_val in holiday_indicator.index:
                                    row_idx = (*series_idx, time_idx_val)
                                    if row_idx in expanded_indicator_df.index:
                                        # Extract scalar value to avoid Series assignment
                                        holiday_value = holiday_indicator.loc[time_idx_val]
                                        if hasattr(holiday_value, 'iloc'):
                                            holiday_value = holiday_value.iloc[0]
                                        expanded_indicator_df.at[row_idx, f'{col_name}_indicator'] = holiday_value
                    else:
                        for time_idx_val in time_idx:
                            if time_idx_val in holiday_indicator.index:
                                if time_idx_val in expanded_indicator_df.index:
                                    # Extract scalar value to avoid Series assignment
                                    holiday_value = holiday_indicator.loc[time_idx_val]
                                    if hasattr(holiday_value, 'iloc'):
                                        holiday_value = holiday_value.iloc[0]
                                    expanded_indicator_df.at[time_idx_val, f'{col_name}_indicator'] = holiday_value
                    
                    # Generate fourier features
                    if self.fourier_features_ is not None:
                        fourier_features_df = self.fourier_features_.transform(date_df)
                        fourier_features_df = fourier_features_df + 1 # Transform from [-1, 1] to [0, 2]
                        
                        if self.expand_column_per_level_ is not None:
                            fourier_features_df = self.expand_column_per_level_.transform(fourier_features_df)
                        
                        # Multiply each fourier feature by the holiday indicator
                        for col in fourier_features_df.columns:
                            combined_col_name = f'{col_name}_{col}'
                            combined_features_list.append(
                                (expanded_indicator_df[f'{col_name}_indicator'].values * fourier_features_df[col].values, combined_col_name)
                            )
            
            # Create combined features DataFrame
            if combined_features_list:
                combined_features_data = np.column_stack([feat[0] for feat in combined_features_list])
                combined_features_columns = [feat[1] for feat in combined_features_list]
                combined_features_df = pd.DataFrame(
                    combined_features_data,
                    index=date_df.index,
                    columns=combined_features_columns
                )
            else:
                combined_features_df = pd.DataFrame(index=date_df.index)
            
            # Fit the linear effect with combined features
            self.linear_effect_.fit(X=combined_features_df, y=y, scale=scale)

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
        # Determine index structure based on univariate/multivariate series
        if self.has_multiple_series_:
            idx_list = self.series_idx_.to_list()
            idx_list = [x if isinstance(x, tuple) else (x,) for x in idx_list]
            idx_tuples = [(*s_idx, t_idx) for s_idx in idx_list for t_idx in fh]
            idx = pd.MultiIndex.from_tuples(idx_tuples, names=self.index_names_)
            time_idx_for_holiday_indicators = fh # For _create_holiday_features, which expects unique time points
        else:
            idx = fh
            time_idx_for_holiday_indicators = fh
            idx_list = []  # Initialize for type checker

        # 1. Create holiday indicator features (binary)
        holiday_indicators_df = _create_holiday_features_df(
            dates_index=time_idx_for_holiday_indicators,
            expanded_holidays_df=self.expanded_holidays_,
            holiday_names=self.holiday_names_
        )

        if self.fourier_terms == 0:
            # When fourier_terms is 0, create a DataFrame with a dummy zero column
            # to avoid empty DataFrame issues in conversion utilities
            empty_features_df = pd.DataFrame(index=idx)
            empty_features_df['dummy_zero'] = 0.0
            return self.linear_effect_.transform(empty_features_df, fh)

        # 2. Create combined features: holiday indicators * fourier features
        combined_features_list = []
        
        for holiday_name in self.holiday_names_:
            # Get the indicator for this holiday
            col_name = f"holiday_{holiday_name}"
            if col_name in holiday_indicators_df.columns:
                holiday_indicator = holiday_indicators_df[col_name]
                
                # Create a DataFrame with the index idx and multiply by holiday indicator
                expanded_indicator_df = pd.DataFrame(index=idx)
                expanded_indicator_df[f'{col_name}_indicator'] = 0.0
                
                # Fill in the indicator values for the appropriate time points
                if self.has_multiple_series_:
                    for i, series_idx in enumerate(idx_list):
                        for j, time_idx in enumerate(fh):
                            if time_idx in holiday_indicator.index:
                                row_idx = (*series_idx, time_idx)
                                if row_idx in expanded_indicator_df.index:
                                    # Extract scalar value to avoid Series assignment
                                    holiday_value = holiday_indicator.loc[time_idx]
                                    if hasattr(holiday_value, 'iloc'):
                                        holiday_value = holiday_value.iloc[0]
                                    expanded_indicator_df.at[row_idx, f'{col_name}_indicator'] = holiday_value
                else:
                    for time_idx in fh:
                        if time_idx in holiday_indicator.index:
                            if time_idx in expanded_indicator_df.index:
                                # Extract scalar value to avoid Series assignment
                                holiday_value = holiday_indicator.loc[time_idx]
                                if hasattr(holiday_value, 'iloc'):
                                    holiday_value = holiday_value.iloc[0]
                                expanded_indicator_df.at[time_idx, f'{col_name}_indicator'] = holiday_value
                
                # Generate fourier features for the entire time range
                date_df = pd.DataFrame(index=idx)
                if self.fourier_features_ is not None:
                    fourier_features_df = self.fourier_features_.transform(date_df)
                    # Transform from [-1, 1] to [0, 2]
                    fourier_features_df = fourier_features_df + 1

                    if self.expand_column_per_level_ is not None:
                        fourier_features_df = self.expand_column_per_level_.transform(fourier_features_df)
                    
                    # Multiply each fourier feature by the holiday indicator to get holiday-specific fourier features
                    for col in fourier_features_df.columns:
                        combined_features_list.append(
                            (expanded_indicator_df[f'{col_name}_indicator'].values * fourier_features_df[col].values, f'{col_name}_{col}')
                        )
        
        # Combine all features into a single DataFrame
        if combined_features_list:
            combined_features_data = np.column_stack([feat[0] for feat in combined_features_list])
            combined_features_columns = [feat[1] for feat in combined_features_list]
            combined_features_df = pd.DataFrame(
                combined_features_data,
                index=idx,
                columns=combined_features_columns
            )
        else:
            combined_features_df = pd.DataFrame(index=idx)
        
        # Use the linear effect transform
        return self.linear_effect_.transform(combined_features_df, fh)

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
        # Use the linear effect to make predictions
        return self.linear_effect_.predict(
            data=data,
            predicted_effects=predicted_effects,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default") -> List[Dict[str, Any]]:  # type: ignore[override]
        """Get test parameters for the holiday effect.
        
        Parameters
        ----------
        parameter_set : str, default "default"
            Parameter set to use.
            
        Returns
        -------
        List[Dict[str, Any]]
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


class BinaryStdHolidayEffect(BinaryHolidayEffect):
    """Holiday effect that uses LinearEffect for standardized prior scaling.
    
    This class inherits from BinaryHolidayEffect to get all the batching functionality,
    but uses LinearEffect for parameter sampling instead of individual holiday parameters.
    This means all holidays will have the same prior scale, ignoring any prior_scale
    values specified in the holidays DataFrame.
    
    This approach is more efficient than BinaryHolidayEffect when you want uniform
    prior scaling across all holidays, as it leverages LinearEffect's vectorized
    parameter sampling with numpyro plates.
    
    Parameters
    ----------
    holidays : pd.DataFrame
        DataFrame with holidays. Must contain 'holiday' and 'ds' columns.
        The 'prior_scale' column will be ignored - all holidays use the same prior.
    country_name : str, optional
        Country name for built-in holidays.
    lower_window : int, default 0
        Number of days before the holiday to include in the holiday effect.
    upper_window : int, default 0  
        Number of days after the holiday to include in the holiday effect.
    prior_scale : float, default 0.1
        Scale of the prior distribution applied uniformly to all holidays.
        Individual prior_scale values in the holidays DataFrame are ignored.
    include_day_of_week : bool, default False
        Whether to include day of week in holiday identifiers.
    daily_splits : int, default 1
        Number of batches to split each day into for granular modeling.
    granularity : str, default 'h'
        Time granularity ('h' for hourly, 'T'/'min' for minutely, etc).
    periods_per_day : Optional[int], default None
        Number of periods per day. Auto-inferred from granularity if None.
    effect_mode : str, default 'additive'
        Either "multiplicative" or "additive".
        
    Notes
    -----
    Unlike BinaryHolidayEffect, this class:
    - Uses LinearEffect for vectorized parameter sampling
    - Ignores individual prior_scale values from holidays DataFrame
    - Applies uniform prior_scale to all holiday effects
    - More efficient for models with many holidays
    """
    
    def __init__(
        self,
        holidays: pd.DataFrame,
        country_name: Optional[str] = None,
        lower_window: int = 0,
        upper_window: int = 0,
        prior_scale: float = 0.1,
        include_day_of_week: bool = False,
        daily_splits: int = 1,
        granularity: str = 'h',
        periods_per_day: Optional[int] = None,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive"
    ):
        # Initialize parent class
        super().__init__(
            holidays=holidays,
            country_name=country_name,
            lower_window=lower_window,
            upper_window=upper_window,
            prior_scale=prior_scale,
            include_day_of_week=include_day_of_week,
            daily_splits=daily_splits,
            granularity=granularity,
            periods_per_day=periods_per_day,
            effect_mode=effect_mode
        )
        
        # Initialize LinearEffect for standardized parameter sampling
        self.linear_effect_ = LinearEffect(
            effect_mode=self.effect_mode,
            prior=dist.Normal(0, self.prior_scale)
        )
    
    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Fit the holiday effect using LinearEffect.
        
        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe.
        X : pd.DataFrame
            The DataFrame to initialize the effect.
        scale : float, optional
            The scale of the timeseries, by default 1.0.
        """
        # Call parent's fit to set up holidays data
        super()._fit(y, X, scale)
        
        if not self.holiday_names_:
            # No holidays to fit
            return
            
        # Create holiday features using the same logic as parent class
        time_idx = X.index.get_level_values(-1) if self.has_multiple_series_ else X.index
        
        if self.daily_splits == 1:
            # Use standard holiday features
            holiday_features_df = _create_holiday_features_df(
                dates_index=time_idx,
                expanded_holidays_df=self.expanded_holidays_,
                holiday_names=self.holiday_names_
            )
        else:
            # Use batched holiday features
            holiday_features_df = _create_batched_holiday_features_df(
                dates_index=time_idx,
                expanded_holidays_df=self.expanded_holidays_,
                holiday_names=self.holiday_names_,
                daily_splits=self.daily_splits,
                periods_per_day=self.periods_per_day
            )
        
        # Expand features for multivariate series if needed
        if self.has_multiple_series_:
            # Create expanded features DataFrame with proper MultiIndex
            expanded_features_df = pd.DataFrame(index=X.index)
            
            # For each series, replicate the holiday features
            series_list = X.index.droplevel(-1).unique().to_list()
            series_list = [x if isinstance(x, tuple) else (x,) for x in series_list]
            
            for col in holiday_features_df.columns:
                expanded_features_df[col] = 0.0
                
                for series_idx in series_list:
                    for time_idx_val in time_idx:
                        if time_idx_val in holiday_features_df.index:
                            row_idx = (*series_idx, time_idx_val)
                            if row_idx in expanded_features_df.index:
                                feature_value = holiday_features_df.loc[time_idx_val, col]
                                # Handle case where feature_value is a Series instead of scalar
                                if isinstance(feature_value, pd.Series):
                                    if len(feature_value) > 0:
                                        feature_value = feature_value.iloc[0]
                                    else:
                                        feature_value = 0.0
                                expanded_features_df.at[row_idx, col] = feature_value
            
            holiday_features_df = expanded_features_df
        
        # Fit the LinearEffect with holiday features
        if holiday_features_df.empty or holiday_features_df.shape[1] == 0:
            # Create a dummy column to avoid empty DataFrame issues
            holiday_features_df = pd.DataFrame(index=holiday_features_df.index)
            holiday_features_df['dummy_zero'] = 0.0
            
        self.linear_effect_.fit(X=holiday_features_df, y=y, scale=scale)
    
    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        """Transform input data using LinearEffect.
        
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
        if not self.holiday_names_:
            # No holidays - create dummy features
            if self.has_multiple_series_:
                idx_list = self.series_idx_.to_list()
                idx_list = [x if isinstance(x, tuple) else (x,) for x in idx_list]
                idx_tuples = [(*s_idx, t_idx) for s_idx in idx_list for t_idx in fh]
                idx = pd.MultiIndex.from_tuples(idx_tuples, names=self.index_names_)
            else:
                idx = fh
                
            dummy_features_df = pd.DataFrame(index=idx)
            dummy_features_df['dummy_zero'] = 0.0
            return self.linear_effect_.transform(dummy_features_df, fh)
        
        # Create holiday features using the same logic as parent class
        time_idx_for_features = fh
        
        if self.daily_splits == 1:
            # Use standard holiday features
            holiday_features_df = _create_holiday_features_df(
                dates_index=time_idx_for_features,
                expanded_holidays_df=self.expanded_holidays_,
                holiday_names=self.holiday_names_
            )
        else:
            # Use batched holiday features
            holiday_features_df = _create_batched_holiday_features_df(
                dates_index=time_idx_for_features,
                expanded_holidays_df=self.expanded_holidays_,
                holiday_names=self.holiday_names_,
                daily_splits=self.daily_splits,
                periods_per_day=self.periods_per_day
            )
        
        # Expand features for multivariate series if needed
        if self.has_multiple_series_:
            idx_list = self.series_idx_.to_list()
            idx_list = [x if isinstance(x, tuple) else (x,) for x in idx_list]
            idx_tuples = [(*s_idx, t_idx) for s_idx in idx_list for t_idx in fh]
            idx = pd.MultiIndex.from_tuples(idx_tuples, names=self.index_names_)
            
            # Create expanded features DataFrame with proper MultiIndex
            expanded_features_df = pd.DataFrame(index=idx)
            
            for col in holiday_features_df.columns:
                expanded_features_df[col] = 0.0
                
                for series_idx in idx_list:
                    for time_idx_val in fh:
                        if time_idx_val in holiday_features_df.index:
                            row_idx = (*series_idx, time_idx_val)
                            if row_idx in expanded_features_df.index:
                                feature_value = holiday_features_df.loc[time_idx_val, col]
                                # Handle case where feature_value is a Series instead of scalar
                                if isinstance(feature_value, pd.Series):
                                    if len(feature_value) > 0:
                                        feature_value = feature_value.iloc[0]
                                    else:
                                        feature_value = 0.0
                                expanded_features_df.at[row_idx, col] = feature_value
                        
            holiday_features_df = expanded_features_df
        
        # Use LinearEffect transform
        return self.linear_effect_.transform(holiday_features_df, fh)
    
    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the holiday effect using LinearEffect.
        
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
        # Delegate to LinearEffect
        return self.linear_effect_.predict(
            data=data,
            predicted_effects=predicted_effects,
        )
    
    @classmethod
    def get_test_params(cls, parameter_set="default") -> List[Dict[str, Any]]:  # type: ignore[override]
        """Get test parameters for BinaryStdHolidayEffect.
        
        Parameters
        ----------
        parameter_set : str, default "default"
            Parameter set to use.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of parameter sets.
        """
        # Create a simple holiday DataFrame for testing
        holidays_df = pd.DataFrame({
            'holiday': ['New Year', 'Christmas'],
            'ds': [f'2023-01-01', f'2023-12-25'],
            'prior_scale': [0.05, 0.15]  # These will be ignored
        })
        
        return [
            {
                'holidays': holidays_df,
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 0.1,  # Uniform prior for all holidays
                'effect_mode': 'additive',
            },
            {
                'holidays': holidays_df,
                'lower_window': 0,
                'upper_window': 0,
                'prior_scale': 0.2,
                'daily_splits': 3,
                'granularity': 'h',
                'effect_mode': 'multiplicative',
            }
        ]


# Module-Level Helper Functions
def _get_dow_name(date: pd.Timestamp) -> str:
    """Get the day of week name for a given date."""
    return date.day_name()

def _get_holiday_years(time_index: pd.Index) -> List[int]:
    """Extract years from the time index."""
    if not isinstance(time_index[0], (datetime, pd.Timestamp)):
        time_index = pd.to_datetime(time_index)
    dt_index: pd.DatetimeIndex = time_index # type: ignore
    return sorted(list(dt_index.year.unique()))

def _get_built_in_holidays_df(country_name: str, years: List[int]) -> pd.DataFrame:
    """Get built-in holidays for the specified country and years."""
    try:
        from holidays import country_holidays # type: ignore
    except ImportError:
        raise ImportError(
            "The 'holidays' package is required for built-in holidays. "
            "Install it with 'pip install holidays'."
        )
    country_holiday_dates = []
    for year in years:
        holidays_dict = country_holidays(country_name, years=year)
        for date, name in holidays_dict.items():
            country_holiday_dates.append({
                'holiday': name,
                'ds': date,
                'country': country_name 
            })
    return pd.DataFrame(country_holiday_dates)

def _expand_single_holiday_row(
    holiday_row: pd.Series, 
    default_lower_window: int, 
    default_upper_window: int, 
    default_prior_scale: float, 
    include_day_of_week: bool
) -> List[Dict[str, Any]]:
    """Expand a single holiday row based on its window settings."""
    expanded_days = []
    holiday_name = holiday_row['holiday']
    holiday_date = holiday_row['ds'] # Should be pd.Timestamp
    
    lower_window = holiday_row.get('lower_window', default_lower_window)
    upper_window = holiday_row.get('upper_window', default_upper_window)
    prior_scale = holiday_row.get('prior_scale', default_prior_scale)
    if prior_scale is None: # Handle cases where prior_scale in DataFrame could be None
        prior_scale = default_prior_scale

    for offset in range(lower_window, upper_window + 1):
        day = holiday_date + pd.Timedelta(days=offset)
        day_of_week_name = _get_dow_name(day)
        
        holiday_id_components = [holiday_name, str(offset)]
        if include_day_of_week:
            holiday_id_components.append(day_of_week_name)
        holiday_id = "_".join(holiday_id_components)

        expanded_days.append({
            'holiday': holiday_id,
            'ds': day.date(), # Store as date object for matching
            'offset': offset,
            'day_of_week': day_of_week_name,
            'prior_scale': prior_scale,
            'country': holiday_row.get('country', 'user_defined')
        })
    return expanded_days

def _preprocess_holiday_df(
    raw_holidays_df: Optional[pd.DataFrame], 
    country_name: Optional[str], 
    time_index_for_years: Optional[pd.Index], 
    default_lower_window: int, 
    default_upper_window: int, 
    default_prior_scale: float, 
    include_day_of_week: bool
) -> pd.DataFrame:
    """Preprocesses holiday data by combining sources and expanding windows."""
    
    processed_holidays_list = []

    if raw_holidays_df is None and country_name is None:
        raise ValueError("Either 'holidays' (raw_holidays_df) or 'country_name' must be provided.")

    current_holidays_df = pd.DataFrame()

    if raw_holidays_df is not None:
        if 'holiday' not in raw_holidays_df.columns or 'ds' not in raw_holidays_df.columns:
            raise ValueError("The user-provided holidays DataFrame must contain 'holiday' and 'ds' columns.")
        current_holidays_df = raw_holidays_df.copy()

    if country_name is not None:
        if time_index_for_years is None and raw_holidays_df is None:
            raise ValueError("time_index_for_years must be provided if using country_name without a base holiday DataFrame to infer years.")
        
        # Infer years from time_index_for_years if available, otherwise from ds column of raw_holidays_df
        years_to_fetch = []
        if time_index_for_years is not None:
            years_to_fetch = _get_holiday_years(time_index_for_years)
        elif not current_holidays_df.empty and pd.api.types.is_datetime64_any_dtype(current_holidays_df['ds']):
            years_to_fetch = sorted(list(pd.to_datetime(current_holidays_df['ds']).dt.year.unique()))
        
        if not years_to_fetch and country_name: # If still no years but country name is given
            raise ValueError("Could not determine years for fetching built-in holidays. Provide time_index_for_years or ensure 'ds' in raw_holidays_df is datetime.")

        built_in_holidays = _get_built_in_holidays_df(country_name, years_to_fetch)
        current_holidays_df = pd.concat([current_holidays_df, built_in_holidays], ignore_index=True)
    
    if current_holidays_df.empty:
        return pd.DataFrame(columns=['holiday', 'ds', 'offset', 'day_of_week', 'prior_scale', 'country'])

    if not pd.api.types.is_datetime64_any_dtype(current_holidays_df['ds']):
        current_holidays_df['ds'] = pd.to_datetime(current_holidays_df['ds'])
    
    # Fill missing window/prior_scale columns with defaults before expansion
    if 'lower_window' not in current_holidays_df.columns:
        current_holidays_df['lower_window'] = default_lower_window
    if 'upper_window' not in current_holidays_df.columns:
        current_holidays_df['upper_window'] = default_upper_window
    if 'prior_scale' not in current_holidays_df.columns:
        current_holidays_df['prior_scale'] = default_prior_scale
    
    # Ensure defaults are applied if values are None (e.g. from concat)
    current_holidays_df['lower_window'] = current_holidays_df['lower_window'].fillna(default_lower_window)
    current_holidays_df['upper_window'] = current_holidays_df['upper_window'].fillna(default_upper_window)
    current_holidays_df['prior_scale'] = current_holidays_df['prior_scale'].fillna(default_prior_scale)
    
    # Expand holidays over windows and include day of week if specified
    for _, holiday_row in current_holidays_df.iterrows():
        holiday_name = holiday_row['holiday']
        holiday_date = holiday_row['ds']
        lower_window = int(holiday_row['lower_window'])
        upper_window = int(holiday_row['upper_window'])
        prior_scale = float(holiday_row['prior_scale'])
        
        # Expand the holiday over its window
        for offset in range(lower_window, upper_window + 1):
            date_in_window = holiday_date + pd.Timedelta(days=offset)
            
            # Create the holiday identifier
            if include_day_of_week:
                day_of_week = _get_dow_name(date_in_window)
                holiday_identifier = f"{holiday_name}_{offset}_{day_of_week}"
            else:
                holiday_identifier = f"{holiday_name}_{offset}"
                
            # Add to the processed holidays list
            processed_holidays_list.append({
                'holiday': holiday_identifier,
                'ds': date_in_window,
                'offset': offset,
                'day_of_week': _get_dow_name(date_in_window),
                'prior_scale': prior_scale,
                'country': country_name if country_name else 'user_provided'
            })
    
    # Convert to DataFrame and return
    if processed_holidays_list:
        return pd.DataFrame(processed_holidays_list)
    else:
        return pd.DataFrame(columns=['holiday', 'ds', 'offset', 'day_of_week', 'prior_scale', 'country'])


def _create_holiday_features_df(
    dates_index: pd.Index, 
    expanded_holidays_df: pd.DataFrame, 
    holiday_names: List[str]
) -> pd.DataFrame:
    """Create holiday feature DataFrame for given dates and holidays.
    
    Parameters
    ----------
    dates_index : pd.Index
        Index of dates to create features for.
    expanded_holidays_df : pd.DataFrame
        DataFrame with expanded holidays containing 'holiday' and 'ds' columns.
    holiday_names : List[str]
        List of unique holiday names.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with holiday features as columns and dates as index.
    """
    # Initialize features DataFrame with zeros
    features_df = pd.DataFrame(index=dates_index)
    
    # Create a column for each holiday
    for holiday_name in holiday_names:
        col_name = f"holiday_{holiday_name}"
        features_df[col_name] = 0.0
        
        # Get all dates for this holiday
        holiday_dates = expanded_holidays_df[expanded_holidays_df['holiday'] == holiday_name]['ds']
        
        # Set indicator to 1 for holiday dates that exist in our index
        for holiday_date in holiday_dates:
            # Convert holiday_date to just the date part for comparison
            holiday_date_only = pd.Timestamp(holiday_date).date()
            
            # Find all timestamps in the index that match this date
            matching_timestamps = [
                ts for ts in features_df.index 
                if pd.Timestamp(ts).date() == holiday_date_only
            ]
            
            # Set indicator to 1 for all matching timestamps
            for ts in matching_timestamps:
                features_df.loc[ts, col_name] = 1.0
    
    return features_df

def _create_batched_holiday_features_df(
    dates_index: pd.Index, 
    expanded_holidays_df: pd.DataFrame, 
    holiday_names: List[str],
    daily_splits: int,
    periods_per_day: int
) -> pd.DataFrame:
    """Create holiday feature DataFrame with batching support for given dates and holidays.
    
    Parameters
    ----------
    dates_index : pd.Index
        Index of dates to create features for.
    expanded_holidays_df : pd.DataFrame
        DataFrame with expanded holidays containing 'holiday' and 'ds' columns.
    holiday_names : List[str]
        List of unique holiday names.
    daily_splits : int
        Number of batches to split each day into.
    periods_per_day : int
        Number of periods per day.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with holiday features as columns and dates as index.
        If daily_splits > 1, creates separate columns for each batch.
    """
    # Initialize features DataFrame with zeros
    features_df = pd.DataFrame(index=dates_index)
    
    if daily_splits == 1:
        # Use the original logic for no batching
        return _create_holiday_features_df(dates_index, expanded_holidays_df, holiday_names)
    
    # Calculate batch size
    batch_size = periods_per_day // daily_splits
    
    # Create a column for each holiday and each batch
    for holiday_name in holiday_names:
        for batch_idx in range(daily_splits):
            col_name = f"holiday_{holiday_name}_batch_{batch_idx}"
            features_df[col_name] = 0.0
        
        # Get all dates for this holiday
        holiday_dates = expanded_holidays_df[expanded_holidays_df['holiday'] == holiday_name]['ds']
        
        # Set indicator to 1 for holiday dates that exist in our index
        for holiday_date in holiday_dates:
            # Convert holiday_date to just the date part for comparison
            holiday_date_only = pd.Timestamp(holiday_date).date()
            
            # Find all timestamps in the index that match this date
            matching_timestamps = [
                ts for ts in features_df.index 
                if pd.Timestamp(ts).date() == holiday_date_only
            ]
            
            # For each matching timestamp, determine which batch it belongs to
            for ts in matching_timestamps:
                timestamp = pd.Timestamp(ts)
                if hasattr(timestamp, 'hour') and hasattr(timestamp, 'minute'):
                    # Calculate period within day based on granularity
                    # Convert timestamp to minutes since start of day
                    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
                    
                    # Calculate period of day based on periods_per_day
                    minutes_per_period = (24 * 60) // periods_per_day
                    period_of_day = minutes_since_midnight // minutes_per_period
                    
                    # Ensure we don't exceed periods_per_day
                    period_of_day = min(period_of_day, periods_per_day - 1)
                    
                    # Determine which batch this period belongs to
                    batch_idx = period_of_day // batch_size
                    # Ensure batch_idx doesn't exceed daily_splits-1
                    batch_idx = min(batch_idx, daily_splits - 1)
                    
                    col_name = f"holiday_{holiday_name}_batch_{batch_idx}"
                    features_df.loc[ts, col_name] = 1.0
                else:
                    # If no time information, assign to first batch
                    col_name = f"holiday_{holiday_name}_batch_0"
                    features_df.loc[ts, col_name] = 1.0
    
    return features_df