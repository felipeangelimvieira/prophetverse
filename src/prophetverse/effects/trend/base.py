"""Module containing the base class for trend models."""

import pandas as pd

from prophetverse.utils.frame_to_array import convert_index_to_days_since_epoch


class TrendEffectMixin:
    """
    Mixin class for trend models.

    Trend models are effects applied to the trend component of a time series.

    Attributes
    ----------
    t_scale: float
        The time scale of the trend model.
    t_start: float
        The starting time of the trend model.
    n_series: int
        The number of series in the time series data.
    """

    _tags = {
        "requires_X": False,
        "hierarchical_prophet_compliant": True,
        "capability:multivariate_input": True,
    }

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1) -> None:
        """Initialize the effect.

        Set the time scale, starting time, and number of series attributes.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.
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

    def _index_to_scaled_timearray(self, idx):
        """
        Convert the index to a scaled time array.

        Parameters
        ----------
        idx: int
            The index to be converted.

        Returns
        -------
        float
            The scaled time array value.
        """
        if idx.nlevels > 1:
            idx = idx.get_level_values(-1).unique()

        t_days = convert_index_to_days_since_epoch(idx)
        return (t_days) / self.t_scale - self.t_start
