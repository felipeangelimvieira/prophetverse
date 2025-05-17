from prophetverse.experimental.budget_optimization.optimizer import (
    BudgetOptimizer,
)
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.experimental.simulate import simulate
from prophetverse.effects import HillEffect, GeometricAdstockEffect, ChainedEffects
import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp


@pytest.fixture
def X():
    """Generate synthetic data for testing."""
    # Create 100 sequential periods starting from 2023-01-01 with daily frequency.
    dates = pd.period_range(start="2023-01-01", periods=100, freq="D")
    # Generate a 100x2 array of random values.
    data = np.random.random((100, 2))
    X = pd.DataFrame(data, columns=["channel1", "channel2"], index=dates)
    return X


@pytest.fixture
def model():

    # Create a Prophetverse model
    model = Prophetverse(
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
    )
    return model


@pytest.fixture
def data(model, X):
    """Generate synthetic data for testing."""
    # Simulate data

    # Fit the model to the data
    components, true_model = simulate(
        model=model.clone(),
        fh=X.index,
        X=X,
        return_model=True,
    )
    return components, X, true_model


def _run_test(optimizer: BudgetOptimizer, components, X, true_model):
    """Run the test for the BudgetOptimizer."""
    # Initialize the optimizer
    X_before = X.copy()
    horizon = X.index[1:]
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

    not_equal = X_opt != X
    assert not_equal.loc[horizon].sum().sum() == len(horizon) * len(X.columns)
    assert not_equal.loc[~not_equal.index.isin(horizon)].sum().sum() == 0

    # assert no nans in optimizer.result_.x
    assert not jnp.isnan(optimizer.result_.x).any()


def test_budget_optimizer_initialization(data):
    """Test the initialization of the BudgetOptimizer."""
    components, X, true_model = data

    for test_params in BudgetOptimizer.get_test_params():
        optimizer = BudgetOptimizer(**test_params)
        _run_test(optimizer, components, X, true_model)
