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

    _tags = {"skip_predict_if_no_match": False, "supports_multivariate": True}

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, scale: float = 1) -> None:
        """Initialize trend model with the timeseries data.

        This method is close to what "fit" is in sktime/sklearn estimators.
        Child classes should implement this method to initialize the model and
        may call super().initialize() to perform common initialization steps.

        Parameters
        ----------
        y: pd.DataFrame
            time series dataframe, may be multiindex
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
