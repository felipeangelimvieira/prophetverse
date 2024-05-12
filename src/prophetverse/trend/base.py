

from abc import ABC, abstractmethod

import pandas as pd

from prophetverse.utils.frame_to_array import convert_index_to_days_since_epoch


class TrendModel(ABC):
    """
    Abstract base class for trend models.

    Attributes:
        t_scale (float): The time scale of the trend model.
        t_start (float): The starting time of the trend model.
        n_series (int): The number of series in the time series data.

    """

    def initialize(self, y: pd.DataFrame) -> None:
        """Initialize trend model with the timeseries data, as an pd.Dataframe.
        This method is close to what "fit" is in sktime/sklearn estimators.
        Child classes should implement this method to initialize the model and
        may call super().initialize() to perform common initialization steps.

        Args:
            y (pd.DataFrame): time series dataframe, may be multiindex

        """

        # Set time scale
        t_days = convert_index_to_days_since_epoch(
            y.index.get_level_values(-1).unique()
        )
        self.t_scale = (t_days[1:] - t_days[:-1]).mean()
        self.t_start = t_days.min() / self.t_scale
        if y.index.nlevels > 1:
            self.n_series = y.index.droplevel(-1).nunique()
        else:
            self.n_series = 1

    @abstractmethod
    def prepare_input_data(self, idx: pd.PeriodIndex) -> dict:
        """
        Returns a dictionary containing the data needed for the trend model.
        All arguments in the signature of compute_trend should be keys in the dictionary.


        For example, given t, a possible implementation would return
        ```python
        {
            "changepoint_matrix": jnp.array([[1, 0], [0, 1]]),
        }
        ```
        
        And compute_trend would be defined as
        
        ```python
        def compute_trend(self, changepoint_matrix):
            ...
        ```
        
        Args:
            idx (pd.PeriodIndex): The index of the time series data.
            
        Returns:
            dict: A dictionary containing the data needed for the trend model.

        """

    @abstractmethod
    def compute_trend(self, **kwargs):
        """Trend model computation. Receives the output of prepare_input_data as keyword arguments.

        Returns:
            jnp.ndarray: array with trend data for each time step and series.
        """
        ...

    def _index_to_scaled_timearray(self, idx):
        """
        Converts the index to a scaled time array.

        Args:
            idx (int): The index to be converted.

        Returns:
            float: The scaled time array value.
        """
        
        t_days = convert_index_to_days_since_epoch(idx)
        return (t_days) / self.t_scale - self.t_start

    def __call__(self, **kwargs):
        return self.compute_trend(**kwargs)

        
        
        