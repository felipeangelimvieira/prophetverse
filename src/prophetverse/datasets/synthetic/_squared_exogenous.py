import numpy as np
import pandas as pd

__all__ = ["load_synthetic_squared_exogenous"]


def _generate_dataset(
    n_periods: int,
    seasonality_period: int,
    trend_slope: float,
    exogenous_range: tuple,
    noise_std: float = 1.0,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate a simple synthetic time series in sktime format.

    The series is composed of seasonality, trend,
    and exogenous variables.

    Parameters
    ----------
    n_periods : int
        Number of time periods to simulate.
    seasonality_period : int
        Period of the seasonal component.
    trend_slope : float
        Slope of the linear trend.
    exogenous_range : tuple
        Range (min, max) for the exogenous variable values.
    noise_std : float, optional
        Standard deviation of the Gaussian noise, by default 1.0.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        y : pd.Series
            Target variable in sktime format with time index.
        X : pd.DataFrame
            Exogenous variables and components in sktime format with time index.

    Examples
    --------
    >>> y, X = generate_sktime_time_series(100, 12, 0.5, (1, 10), 0.5, seed=42)
    >>> y.head()
    time
    0    0.838422
    1    1.488498
    2    2.230748
    3    2.930336
    4    3.724452
    Name: target, dtype: float64
    >>> X.head()
       seasonality     trend  exogenous     noise
    time
    0      0.000000  0.000000   5.749081  0.211731
    1      0.258819  0.500000   6.901429  0.326080
    2      0.500000  1.000000   6.463987  0.460959
    3      0.707107  1.500000   5.197317  0.676962
    4      0.866025  2.000000   3.312037  0.546416
    """
    rng = np.random.default_rng(seed)

    # Time index
    time_index = pd.period_range(
        start="2010-01-01", freq="D", periods=n_periods, name="time"
    )

    # Seasonal component
    seasonality = np.sin(2 * np.pi * np.arange(n_periods) / seasonality_period)

    _t = np.arange(n_periods)
    _t = _t - _t.mean()
    _t = _t / n_periods * 20
    # Linear trend
    trend = trend_slope / (1 + np.exp(-_t))

    # Exogenous variable
    exogenous = rng.uniform(*exogenous_range, size=n_periods)

    # Logarithmic effect of exogenous variable
    exog_effect = 2 * (exogenous - 5) ** 2  # Adding 1 to avoid log(0)

    # Noise
    noise = rng.normal(scale=noise_std, size=n_periods)

    # Target variable
    target = seasonality + trend + exog_effect + noise

    # Construct y and X
    y = pd.Series(data=target, index=time_index, name="target").to_frame()
    X = pd.DataFrame(
        data={
            "exogenous": exogenous,
        },
        index=time_index,
    )

    return y, X


def load_synthetic_squared_exogenous():
    """Load the synthetic log exogenous dataset.

    This dataset is just for documentation purposes.

    Returns
    -------
    pd.DataFrame
        The synthetic target variable
    pd.DataFrame
        The synthetic exogenous variable

    """
    return _generate_dataset(
        n_periods=700,
        seasonality_period=365.25,
        trend_slope=10,
        exogenous_range=(1, 10),
        noise_std=2,
        seed=42,
    )
