"""Loaders for the datasets used in the examples."""

from pathlib import Path
from typing import Union

import pandas as pd
from sktime.datasets import load_forecastingdata
from sktime.transformations.hierarchical.aggregate import Aggregator

DATASET_MODULE_PATH = Path(__file__).parent


def load_tourism(groupby: Union[list[str], str] = "Region"):
    """
    Load the tourism dataset from Athanasopoulos et al. (2011).

    Parameters
    ----------
    groupby : list[str] or str, optional
        The columns to group by. Defaults to "Region".

    Returns
    -------
    pd.DataFrame
        The tourism dataset.

    See Also
    --------
    [1] Athanasopoulos, G., Hyndman, R. J., Song, H., & Wu, D. C. (2011).
    The tourism forecasting competition. International Journal of Forecasting,
    27(3), 822-844.
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    groupby = [g.lower() for g in groupby]

    # Verify if the groupby columns are valid
    valid_columns = ["region", "purpose", "state", "quarter"]
    diff = set(groupby) - set(valid_columns)
    assert not diff, f"Invalid columns: {diff}"

    data = pd.read_csv(DATASET_MODULE_PATH / "raw/tourism.csv")
    data["Quarter"] = pd.PeriodIndex(data["Quarter"], freq="Q")
    data = data.set_index(["Region", "Purpose", "State", "Quarter"])[["Trips"]]
    data = data.sort_index()

    idxs_to_groupby = [g.capitalize() for g in groupby]

    y = (
        Aggregator(flatten_single_levels=False)
        .fit_transform(data)
        .groupby(level=[*idxs_to_groupby, -1])
        .sum()
    )

    return y


def load_peyton_manning():
    """Load the Peyton Manning dataset (used in Prophet's documentation).

    Returns
    -------
    pd.DataFrame
        the Peyton Manning dataset.
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/"
        "examples/example_wp_log_peyton_manning.csv"
    )
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("D")
    return df.set_index("ds")


def load_tensorflow_github_stars():
    """Load the TensorFlow GitHub stars dataset.

    Returns
    -------
    pd.DataFrame
        The TensorFlow GitHub stars dataset.
    """
    df = pd.read_csv(
        DATASET_MODULE_PATH / "raw/tensorflow_tensorflow-stars-history.csv"
    )
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y").dt.to_period("D")
    df = df.set_index("date")
    df = df.sort_index()
    y = df[["day-stars"]]

    return y


def load_pedestrian_count():
    """Load the pedestrian count dataset.

    Returns
    -------
    pd.DataFrame
        The pedestrian count dataset.
    """

    def _parse_data(df):

        dfs = []
        # iterrows
        for _, row in df.iterrows():

            _df = pd.DataFrame(
                data={
                    "pedestrian_count": row["series_value"],
                    "timestamp": pd.period_range(
                        row["start_timestamp"],
                        periods=len(row["series_value"]),
                        freq="H",
                    ),
                },
            )
            _df["series_name"] = row["series_name"]
            dfs.append(_df)

        return pd.concat(dfs).set_index(["series_name", "timestamp"])

    df, _ = load_forecastingdata("pedestrian_counts_dataset")
    return _parse_data(df)
