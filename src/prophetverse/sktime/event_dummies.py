from typing import Optional
import pandas as pd
from sktime.transformations.base import BaseTransformer
from sklearn.utils.validation import check_is_fitted


class EventsDummyTransformer(BaseTransformer):
    """
    Create holiday dummy variables from an events dataframe in fbprophet format.

    The expected events dataframe must have the following columns:
        - event_name: the name of the holiday event.
        - ds: the date of the event (datetime convertible).
        - lower_window: integer offset (in days) indicating how many days
          before the event the effect starts.
        - upper_window: integer offset (in days) indicating how many days
          after the event the effect ends.

    The transformer produces dummy variables such that for each event row the
    effective dates are all dates in the range ds + lower_window to ds + upper_window.
    If dummy_by_window is False, a single dummy column is created per event (with an
    optional prefix). If True, a separate dummy is created for each offset within the
    window (with the column name reflecting the offset).

    During transform, if X is None, the stored dummy dataframe (with a daily date
    range spanning from the earliest to the latest effective holiday date) is returned.
    Otherwise, a left join is performed on X so that only the dates available in X appear,
    while preserving the index dtype of X.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame of holiday events with columns: event_name, ds, lower_window, upper_window.
    prefix : Optional[str], default=None
        Optional prefix to add to the dummy column names.
    dummy_by_window : bool, default=False
        If True, creates a separate dummy for each window offset (each day in the holiday window).
        Otherwise, creates one dummy column per event.

    Attributes
    ----------
    events_df_ : pd.DataFrame
        Copy of events_df with ds converted to datetime.
    _dummy_df_ : pd.DataFrame
        Dummy dataframe covering the full date range from the earliest to the latest effective date.
    """

    _tags = {
        "fit_is_empty": False,
        "requires_X": False,
        "remember_data": False,
        "X_inner_mtype": "pd.DataFrame",
        "transform-returns-same-time-index": True,
    }

    def __init__(
        self,
        events_df: pd.DataFrame,
        prefix: Optional[str] = None,
        dummy_by_window: bool = False,
    ):
        self.events_df = events_df
        self.prefix = prefix
        self.dummy_by_window = dummy_by_window
        super().__init__()

    def _fit(self, X, y=None):
        """
        Fit the transformer.

        Converts the 'ds' column to datetime and precomputes the dummy DataFrame,
        storing it in self._dummy_df_.

        Parameters
        ----------
        X : pd.DataFrame or None
            Ignored.
        y : Ignored.

        Returns
        -------
        self : reference to self.
        """
        self.events_df_ = self.events_df.copy()
        self.events_df_["ds"] = pd.to_datetime(self.events_df_["ds"])

        # Build the dummy data from events.
        rows = []
        for _, row in self.events_df_.iterrows():
            event = row["event_name"]
            ds = row["ds"]
            lower = int(row["lower_window"])
            upper = int(row["upper_window"])
            if self.dummy_by_window:
                for offset in range(lower, upper + 1):
                    effective_date = ds + pd.Timedelta(days=offset)
                    col = (
                        f"{self.prefix}_{event}_offset_{offset}"
                        if self.prefix
                        else f"{event}_offset_{offset}"
                    )
                    rows.append((effective_date, col))
            else:
                col = f"{self.prefix}_{event}" if self.prefix else f"{event}"
                for offset in range(lower, upper + 1):
                    effective_date = ds + pd.Timedelta(days=offset)
                    rows.append((effective_date, col))
        if not rows:
            raise ValueError("No events found to transform.")

        temp_df = pd.DataFrame(rows, columns=["date", "column"])
        temp_df["value"] = 1
        dummy_df = temp_df.pivot_table(
            index="date", columns="column", values="value", aggfunc="max", fill_value=0
        )
        # Ensure dummy_df has a DatetimeIndex.
        dummy_df.index = pd.to_datetime(dummy_df.index)
        # Reindex to cover the full range.
        full_index = pd.date_range(
            start=dummy_df.index.min(), end=dummy_df.index.max(), freq="D"
        )
        self._dummy_df_ = dummy_df.reindex(full_index, fill_value=0)
        return self

    def _transform(self, X, y=None):
        """
        Transform the events into holiday dummy variables.

        If X is None, returns the stored dummy DataFrame.
        Otherwise, it left joins X with the dummy DataFrame so that only the dates
        in X are kept. The output index is converted back to the same type as X.index.

        Parameters
        ----------
        X : pd.DataFrame or None
            DataFrame whose index is used to subset the dummy DataFrame.
        y : Ignored.

        Returns
        -------
        X_transformed : pd.DataFrame
            Left-joined DataFrame of X and dummy variables.
        """
        check_is_fitted(self, attributes=["_dummy_df_"])

        dummy_df = self._dummy_df_.copy()
        if X is None:
            return dummy_df
        else:
            # Preserve the original index type of X.
            # First, determine an "output index" from X.index.
            if isinstance(X.index, pd.PeriodIndex):
                # Convert PeriodIndex to timestamp for reindexing dummy_df.
                ts_index = X.index.to_timestamp()
                dummy_reindexed = dummy_df.reindex(ts_index, fill_value=0)
                # Convert back to PeriodIndex with same frequency.
                out_index = pd.PeriodIndex(dummy_reindexed.index, freq=X.index.freq)
                dummy_reindexed.index = out_index
            elif isinstance(X.index, (pd.DatetimeIndex, pd.RangeIndex)):
                ts_index = pd.to_datetime(X.index)
                dummy_reindexed = dummy_df.reindex(ts_index, fill_value=0)
                out_index = dummy_reindexed.index
            else:
                # For any other index type, attempt conversion to DatetimeIndex.
                ts_index = pd.to_datetime(X.index, errors="coerce")
                if ts_index.isnull().any():
                    raise ValueError(
                        "X.index could not be converted to a DatetimeIndex."
                    )
                dummy_reindexed = dummy_df.reindex(ts_index, fill_value=0)
                out_index = dummy_reindexed.index

            # Now, perform a left join: we keep X's columns and add dummy columns.
            X_copy = X.copy()
            X_copy = X_copy.reindex(out_index)
            X_joined = X_copy.join(dummy_reindexed, how="left")
            X_joined.fillna(0, inplace=True)
            # Ensure the output index has the same dtype as the input.
            X_joined.index = X.index
            return X_joined

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the EventsDummyTransformer.

        Returns
        -------
        params : list of dict
            A list of parameter dictionaries.
        """
        params = []
        test_events = pd.DataFrame(
            {
                "event_name": ["holiday1", "holiday2"],
                "ds": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")],
                "lower_window": [-1, 0],
                "upper_window": [1, 2],
            }
        )
        params.append(
            {"events_df": test_events, "prefix": "test", "dummy_by_window": False}
        )

        test_events = pd.DataFrame(
            {
                "event_name": ["holiday3", "holiday3"],
                "ds": [pd.Timestamp("2020-02-01"), pd.Timestamp("2021-02-10")],
                "lower_window": [-1, 0],
                "upper_window": [1, 2],
            }
        )
        params.append(
            {"events_df": test_events, "prefix": "test", "dummy_by_window": True}
        )
        return params
