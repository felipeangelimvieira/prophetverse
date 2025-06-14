import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophetverse.datasets.loaders import load_tourism
import jax.numpy as jnp
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.effects.trend import PiecewiseLinearTrend, PiecewiseLogisticTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.utils import no_input_columns
from prophetverse.effects.linear import PanelHierarchicalLinearEffect

# Load data
y = load_tourism(groupby="Purpose")


# Helper function for plotting
LEVELS = y.index.get_level_values(0).unique()


def plot_preds(y=None, preds={}, axs=None):
    if axs is None:
        fig, axs = plt.subplots(
            figsize=(12, 8), nrows=int(np.ceil(len(LEVELS) / 2)), ncols=2
        )
    ax_generator = iter(axs.flatten())
    for level in LEVELS:
        ax = next(ax_generator)
        if y is not None:
            y.loc[level].iloc[:, 0].rename("Observation").plot(
                ax=ax, label="truth", color="black"
            )
        for name, _preds in preds.items():
            _preds.loc[level].iloc[:, 0].rename(name).plot(ax=ax, legend=True)
        ax.set_title(level)
    plt.tight_layout()
    return ax


# Fit univariate Prophetverse model
model = Prophetverse(
    trend=PiecewiseLogisticTrend(
        changepoint_prior_scale=0.1,
        changepoint_interval=8,
        changepoint_range=-8,
    ),
    exogenous_effects=[
        (
            "seasonality",
            LinearFourierSeasonality(
                sp_list=["Y"],
                fourier_terms_list=[1],
                freq="Q",
                prior_scale=0.1,
                effect_mode="multiplicative",
            ),
            no_input_columns,
        )
    ],
    inference_engine=MAPInferenceEngine(),
)


# Fit hierarchical Bayesian model
model_hier = model.clone()
model_hier.set_params(
    seasonality__linear_effect=PanelHierarchicalLinearEffect(), panel_model=True
)
model_hier.fit(y=y)
