import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophetverse.effects.base import BaseEffect


def plot_prior_predictive(
    effect: BaseEffect,
    X=None,
    y=None,
    predicted_effects=None,
    num_samples=1000,
    coverage=0.95,
    matplotlib_kwargs=None,
    mode="time",
):

    if (X is not None) and (X.index.nlevels > 1):  # pragma : no cover
        raise ValueError("This utility does not work with panel data yet.")

    if (y is not None) and (y.index.nlevels > 1):
        raise ValueError("This utility does not work with panel data yet.")

    samples = effect.sample_prior(
        X=X,
        y=y,
        num_samples=num_samples,
        predicted_effects=predicted_effects,
        as_pandas=True,
        seed=42,
    ).unstack(level=0)

    idx = samples.index.get_level_values(-1).unique()

    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()

    alpha = (1 - coverage) / 2
    quantiles = [alpha, 1 - alpha]

    prior_samples = samples.quantile(quantiles, axis=1).T

    matplotlib_kwargs = {} if matplotlib_kwargs is None else matplotlib_kwargs

    fig, ax = plt.subplots(**matplotlib_kwargs)

    if mode == "time":
        _x = idx
        argsort = np.arange(len(idx))
    else:
        _x = X[mode].values.flatten()
        argsort = np.argsort(_x)

    ax.fill_between(
        x=_x[argsort],
        y1=prior_samples[quantiles[0]].values.flatten()[argsort],
        y2=prior_samples[quantiles[1]].values.flatten()[argsort],
        alpha=0.2,
    )

    ax.plot(_x[argsort], samples.mean(axis=1).iloc[argsort])

    return fig, ax
