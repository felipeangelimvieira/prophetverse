import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp

from prophetverse.experimental.budget_optimization.optimizer import BudgetOptimizer
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.effects import HillEffect, LinearEffect
from prophetverse.engine.prior import PriorPredictiveInferenceEngine

DATASET_TYPES = ["simple", "panel"]
BROADCAST_MODES = ["estimator", "effect"]
OPTIMIZER_CFGS = BudgetOptimizer.get_test_params()


def pytest_generate_tests(metafunc):
    """Param-drive any test function that asks for our fixture names."""
    if "dataset_type" in metafunc.fixturenames:
        metafunc.parametrize(
            "dataset_type",
            DATASET_TYPES,
            ids=[f"dataset:{x}" for x in DATASET_TYPES],
        )

    if "broadcast_mode" in metafunc.fixturenames:
        metafunc.parametrize(
            "broadcast_mode",
            BROADCAST_MODES,
            ids=[f"broadcast:{x}" for x in BROADCAST_MODES],
        )

    if "optimizer_cfg" in metafunc.fixturenames:
        metafunc.parametrize(
            "optimizer_cfg",
            OPTIMIZER_CFGS,
        )


@pytest.fixture
def X(dataset_type):
    """Synthetic exogenous features (simple 1-panel vs 2-panel)."""
    rng = np.random.default_rng(42)
    dates = pd.period_range("2023-01-01", periods=100, freq="D")
    base = pd.DataFrame(
        rng.random((100, 2)), columns=["channel1", "channel2"], index=dates
    )

    if dataset_type == "simple":
        return base
    # dataset_type == "panel"
    return pd.concat({"a": base, "b": base + 1}, axis=0)


@pytest.fixture
def model(broadcast_mode):
    """Flat-trend Prophetverse with two exogenous-effect pipelines."""
    return Prophetverse(
        trend="flat",
        exogenous_effects=[
            ("channel1", HillEffect(), "channel1"),
            ("channel2", LinearEffect(), "channel2"),
        ],
        inference_engine=PriorPredictiveInferenceEngine(num_samples=10),
        broadcast_mode=broadcast_mode,  # "estimator" or "effects"
    )


@pytest.fixture
def dataset(model, X):
    """(fitted_model, X, y) tuple ready for optimisation tests."""
    rng = np.random.default_rng(42)
    y = pd.DataFrame(index=X.index, data={"value": rng.normal(X.shape[0])})
    fitted = model.fit(X=X, y=y)
    return fitted, X, y


class TestBudgetOptimizer:
    """Cross-product of dataset_type × broadcast_mode × optimizer_cfg."""

    def test_budget_optimizer(self, optimizer_cfg, dataset):
        fitted_model, X, _ = dataset
        optimizer = BudgetOptimizer(**optimizer_cfg)
        _assert_optimization_was_successful(optimizer, X, fitted_model)


def _assert_optimization_was_successful(optimizer, X, true_model):
    X_before = X.copy()
    horizon = X.index.get_level_values(-1).unique()[1:]

    X_opt = optimizer.optimize(
        model=true_model,
        X=X,
        horizon=horizon,
        columns=X.columns.tolist(),
    )

    assert X_before.equals(X)

    assert not X_opt.equals(X)
    assert X_opt.shape == X.shape
    assert X_opt.index.equals(X.index)
    assert X_opt.columns.equals(X.columns)

    diff = X_opt != X
    mask = X_opt.index.get_level_values(-1).isin(horizon)
    assert diff.loc[mask].values.sum() == mask.sum() * len(X.columns)
    assert diff.loc[~mask].values.sum() == 0

    assert not jnp.isnan(optimizer.result_.x).any()
