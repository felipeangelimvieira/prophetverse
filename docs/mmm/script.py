import warnings

warnings.simplefilter(action="ignore")

import numpyro

numpyro.enable_x64()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

from prophetverse.datasets._mmm.dataset1 import get_dataset

y, X, lift_tests, true_components, model = get_dataset()


def plot_spend_comparison(
    X_baseline,
    X_optimized,
    channels,
    indexer,
    *,
    baseline_title="Baseline Spend: Pre-Optimization",
    optimized_title="Optimized Spend: Maximizing KPI",
    figsize=(12, 6),
):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    X_baseline.loc[indexer, channels].plot(ax=ax[0], linewidth=2)
    X_optimized.loc[indexer, channels].plot(ax=ax[1], linewidth=2, linestyle="--")
    ax[0].set_title(baseline_title, fontsize=14, weight="bold")
    ax[1].set_title(optimized_title, fontsize=14, weight="bold")
    for a in ax:
        a.set_ylabel("Spend")
        a.set_xlabel("Date")
        a.legend(loc="upper right", frameon=True)
        a.grid(axis="x", visible=False)
        a.grid(axis="y", linestyle="--", alpha=0.7)
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    y_max = max(
        X_baseline.loc[indexer, channels].max().max(),
        X_optimized.loc[indexer, channels].max().max(),
    )
    for a in ax:
        a.set_ylim(0, y_max * 1.05)
    plt.tight_layout()
    return fig, ax


from prophetverse.experimental.budget_optimization import (
    BudgetOptimizer,
    SharedBudgetConstraint,
    MaximizeKPI,
)

budget_optimizer = BudgetOptimizer(
    objective=MaximizeKPI(),
    constraints=[SharedBudgetConstraint()],
    options={"disp": True},
)

horizon = pd.period_range("2004-01-01", "2004-12-31", freq="D")

X_opt = budget_optimizer.optimize(
    model=model,
    X=X,
    horizon=horizon,
    columns=["ad_spend_search", "ad_spend_social_media"],
)
