import numpy as np
import pandas as pd


def load_composite_effect_example():
    """
    Load a synthetic time series with a composite effect.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the time series data.
    """
    rng = np.random.default_rng(0)
    timeindex = pd.period_range(
        start="2010-01-01", freq="D", periods=365 * 7, name="time"
    )

    t = np.arange(len(timeindex))

    w = np.ones(100) / 100
    trend = np.ones(len(t)) * t / 20 + 10

    seasonality = (
        np.sin(2 * np.pi * t / 365.25) * 0.7
        + np.sin(2 * np.pi * t / 365.25 * 2) * 1
        # + np.sin(2 * np.pi * t / 365.25 * 3) * 0.5
        # + np.sin(2 * np.pi * t / 365.25 * 4) * 0.5
    ) * 0.8 + 1

    exog = np.clip(rng.normal(0.1, 1, size=len(t)), 0, None)
    # rolling mean
    w = np.ones(15) / 15
    exog = np.convolve(exog, w, mode="same")
    exog -= np.min(exog)
    exog_effect = exog * 0.5
    noise = rng.normal(0, 0.1, size=len(t))
    y = pd.DataFrame(
        data={
            "target": trend * (1 + exog_effect + seasonality + noise)
            + trend * exog * (seasonality - seasonality.min() + 1) * 2
        },
        index=timeindex,
    )

    X = pd.DataFrame(
        {
            "investment": exog,
        },
        index=timeindex,
    )
    return y, X
