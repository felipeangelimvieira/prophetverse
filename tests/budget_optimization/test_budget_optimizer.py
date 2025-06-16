import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp

from prophetverse.experimental.budget_optimization.optimizer import BudgetOptimizer
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.experimental.simulate import simulate
from prophetverse.effects import HillEffect, GeometricAdstockEffect, ChainedEffects
from prophetverse.engine import MCMCInferenceEngine


@pytest.fixture
def X():
    """Generate synthetic exogenous features."""
    dates = pd.period_range(start="2023-01-01", periods=100, freq="D")
    data = np.random.random((100, 2))
    return pd.DataFrame(data, columns=["channel1", "channel2"], index=dates)


@pytest.fixture
def model():
    """A flat-trend Prophetverse with two exogenous effect pipelines."""
    return Prophetverse(
        trend="flat",
        exogenous_effects=[
            ("channel1", HillEffect(), "channel1"),
            (
                "channel2",
                ChainedEffects(
                    steps=[
                        ("adstock", GeometricAdstockEffect()),
                        ("hill", HillEffect()),
                    ]
                ),
                "channel2",
            ),
        ],
        inference_engine=MCMCInferenceEngine(num_warmup=10, num_samples=40),
    )


@pytest.fixture(params=["simple", "panel"])
def dataset(request, model, X):
    """
    Parametrized data fixture:
    - “simple” is just one series of length 100
    - “panel” has two panels (“a” and “b”), each of length 100
    """
    if request.param == "simple":
        components, true_model = simulate(
            model=model.clone(),
            fh=X.index,
            X=X,
            return_model=True,
        )
        return components, X, true_model

    # else: panel
    X_panel = pd.concat(
        {"a": X, "b": X + 1},
        axis=0,
    )
    components, true_model = simulate(
        model=model.clone(),
        fh=X_panel.index.get_level_values(-1).unique(),
        X=X_panel,
        return_model=True,
    )
    return components, X_panel, true_model


def _assert_optimization_was_successful(
    optimizer: BudgetOptimizer, components, X, true_model
):
    """Inner helper to keep our test body lean."""
    X_before = X.copy()
    horizon = X.index.get_level_values(-1).unique()[1:]

    X_opt = optimizer.optimize(
        model=true_model,
        X=X,
        horizon=horizon,
        columns=X.columns.tolist(),
    )

    # original X must be untouched, optimized X must differ on the forecast window only
    assert X_before.equals(X)
    assert not X_opt.equals(X)
    assert X_opt.shape == X.shape
    assert X_opt.index.equals(X.index)
    assert X_opt.columns.equals(X.columns)

    # all entries in the horizon should have changed, none outside it
    diff = X_opt != X

    mask = X_opt.index.get_level_values(-1).isin(horizon)
    assert diff.loc[mask].sum().sum() == sum(mask) * len(X.columns)
    assert diff.loc[~mask].sum().sum() == 0

    # final sanity check: no NaNs in the optimizer’s solution
    assert not jnp.isnan(optimizer.result_.x).any()


@pytest.mark.parametrize("test_params", BudgetOptimizer.get_test_params())
def test_budget_optimizer_initialization(test_params, dataset):
    """
    For each optimizer‐constructor config and each dataset scenario
    (simple vs panel), check that:
      - the input is not mutated,
      - the optimized matrix has the right shape & index,
      - only the forecast‐window rows change,
      - the solver actually converged (no NaNs).
    """
    components, X, true_model = dataset
    optimizer = BudgetOptimizer(**test_params)
    _assert_optimization_was_successful(optimizer, components, X, true_model)
