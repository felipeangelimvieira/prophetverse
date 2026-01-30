"""Tests for effects with budget optimization capability.

This module tests that specific effect combinations work correctly
with the BudgetOptimizer.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist

from prophetverse.budget_optimization import (
    BudgetOptimizer,
    MaximizeKPI,
    TotalBudgetConstraint,
)
from prophetverse.effects import (
    ChainedEffects,
    GeometricAdstockEffect,
    HillEffect,
    LinearEffect,
    LogEffect,
    MichaelisMentenEffect,
    WeibullAdstockEffect,
)
from prophetverse.engine.prior import PriorPredictiveInferenceEngine
from prophetverse.sktime.univariate import Prophetverse

RAND_SEED = 42


def _create_simple_dataset(n_timepoints=100):
    """Create a simple dataset for testing.

    Uses a single column to avoid issues with default effects being created
    for unassigned columns.
    """
    dates = pd.period_range("2023-01-01", periods=n_timepoints, freq="D")
    X = pd.DataFrame(
        np.linspace(0.1, 10, num=n_timepoints).reshape((n_timepoints, 1)),
        columns=["channel1"],
        index=dates,
    )
    y = pd.DataFrame(
        {"value": np.arange(n_timepoints, dtype=float)},
        index=dates,
    )
    return X, y


def _create_panel_dataset(n_timepoints=100, n_series=2):
    """Create a panel dataset for testing.

    Uses a single column to avoid issues with default effects being created
    for unassigned columns.
    """
    dates = pd.period_range("2023-01-01", periods=n_timepoints, freq="D")

    dfs_X = []
    dfs_y = []
    for i in range(n_series):
        X_i = pd.DataFrame(
            np.linspace(0.1 + i, 10 + i, num=n_timepoints).reshape((n_timepoints, 1)),
            columns=["channel1"],
            index=dates,
        )
        y_i = pd.DataFrame(
            {"value": np.arange(n_timepoints, dtype=float) + i * 10},
            index=dates,
        )
        dfs_X.append(X_i)
        dfs_y.append(y_i)

    X = pd.concat({f"series_{i}": df for i, df in enumerate(dfs_X)}, axis=0)
    y = pd.concat({f"series_{i}": df for i, df in enumerate(dfs_y)}, axis=0)
    return X, y


# Define specific effect combinations for testing
EFFECT_TEST_CASES = [
    # Simple effects
    pytest.param(
        LinearEffect(prior=dist.Normal(0, 1), effect_mode="additive"),
        id="Linear",
    ),
    pytest.param(
        LogEffect(effect_mode="additive"),
        id="Log",
    ),
    pytest.param(
        HillEffect(
            half_max_prior=dist.HalfNormal(1),
            slope_prior=dist.HalfNormal(1),
            max_effect_prior=dist.HalfNormal(1),
            effect_mode="additive",
        ),
        id="Hill",
    ),
    pytest.param(
        MichaelisMentenEffect(effect_mode="additive"),
        id="MichaelisMenten",
    ),
    # Adstock effects
    pytest.param(
        GeometricAdstockEffect(normalize=True),
        id="GeometricAdstock",
    ),
    pytest.param(
        GeometricAdstockEffect(normalize=False),
        id="GeometricAdstock-unnormalized",
    ),
    pytest.param(
        WeibullAdstockEffect(max_lag=5, initial_history=0.0),
        id="WeibullAdstock",
    ),
    # Chained effects: Adstock + Saturation
    pytest.param(
        ChainedEffects(
            steps=[
                ("adstock", GeometricAdstockEffect(normalize=True)),
                (
                    "saturation",
                    HillEffect(
                        half_max_prior=dist.HalfNormal(1),
                        slope_prior=dist.HalfNormal(1),
                        max_effect_prior=dist.HalfNormal(1),
                        effect_mode="additive",
                    ),
                ),
            ]
        ),
        id="ChainedAdstock-Hill",
    ),
    pytest.param(
        ChainedEffects(
            steps=[
                ("adstock", GeometricAdstockEffect(normalize=True)),
                ("saturation", MichaelisMentenEffect(effect_mode="additive")),
            ]
        ),
        id="ChainedAdstock-MichaelisMenten",
    ),
    pytest.param(
        ChainedEffects(
            steps=[
                ("adstock", WeibullAdstockEffect(max_lag=5, initial_history=0.0)),
                (
                    "saturation",
                    HillEffect(
                        half_max_prior=dist.HalfNormal(1),
                        slope_prior=dist.HalfNormal(1),
                        max_effect_prior=dist.HalfNormal(1),
                        effect_mode="additive",
                    ),
                ),
            ]
        ),
        id="ChainedWeibullAdstock-Hill",
    ),
]

# Effects that have known issues with budget optimization but pass gradient flow tests
# These are separated so we can mark only the budget optimization tests as xfail
EFFECT_TEST_CASES_WITH_KNOWN_ISSUES = [
    # Chained effects: Saturation + Adstock (reversed order)
    # Note: These combinations currently have issues with budget optimization
    pytest.param(
        ChainedEffects(
            steps=[
                (
                    "saturation",
                    HillEffect(
                        half_max_prior=dist.HalfNormal(1),
                        slope_prior=dist.HalfNormal(1),
                        max_effect_prior=dist.HalfNormal(1),
                        effect_mode="additive",
                    ),
                ),
                ("adstock", GeometricAdstockEffect(normalize=True)),
            ]
        ),
        id="ChainedHill-Adstock",
        marks=pytest.mark.xfail(
            reason="Saturation->Adstock order has gradient flow issues in budget optimization"
        ),
    ),
    # Chained effects: Linear + Adstock
    # Note: Linear->Adstock order has gradient flow issues in budget optimization
    pytest.param(
        ChainedEffects(
            steps=[
                (
                    "linear",
                    LinearEffect(prior=dist.Normal(0, 1), effect_mode="additive"),
                ),
                ("adstock", GeometricAdstockEffect(normalize=True)),
            ]
        ),
        id="ChainedLinear-Adstock",
        marks=pytest.mark.xfail(
            reason="Linear->Adstock order has gradient flow issues in budget optimization"
        ),
    ),
]

# All effects for all tests (budget optimization tests will xfail for known issues)
ALL_EFFECT_TEST_CASES = EFFECT_TEST_CASES + EFFECT_TEST_CASES_WITH_KNOWN_ISSUES


class TestBudgetOptimizationEffects:
    """Test that specific effect combinations work with BudgetOptimizer."""

    @pytest.mark.parametrize("effect_instance", ALL_EFFECT_TEST_CASES)
    def test_budget_optimization_simple_dataset(self, effect_instance):
        """Test budget optimization works with effect on a simple dataset."""
        X, y = _create_simple_dataset(n_timepoints=50)

        # Clone the effect to ensure fresh state
        effect = effect_instance.clone()

        # Create model with the effect
        model = Prophetverse(
            trend="flat",
            exogenous_effects=[
                ("channel1", effect, "channel1"),
            ],
            inference_engine=PriorPredictiveInferenceEngine(num_samples=5),
        )

        # Fit the model
        model.fit(X=X, y=y)

        # Create budget optimizer
        budget_optimizer = BudgetOptimizer(
            objective=MaximizeKPI(),
            constraints=[TotalBudgetConstraint()],
            options={"maxiter": 10},  # Small number of iterations for testing
        )

        # Define optimization horizon
        horizon = X.index[-10:]

        # Run optimization
        X_opt = budget_optimizer.optimize(
            model=model,
            X=X,
            horizon=horizon,
            columns=["channel1"],
        )

        # Assertions
        assert X_opt is not None
        assert X_opt.shape == X.shape
        assert X_opt.index.equals(X.index)
        assert X_opt.columns.equals(X.columns)

        # Check that gradients are non-zero (budget optimization requires gradients)
        # Note: We don't assert that values changed because the optimizer might
        # already be at the optimal point given constraints, especially for
        # simple linear effects.
        jac_values = budget_optimizer.jac_(budget_optimizer.x0_, budget_optimizer)
        assert jnp.any(
            jnp.abs(jac_values) > 1e-10
        ), f"Gradients are all zero for {type(effect_instance).__name__}"

    @pytest.mark.parametrize("effect_instance", ALL_EFFECT_TEST_CASES)
    def test_budget_optimization_panel_dataset(self, effect_instance):
        """Test budget optimization works with effect on a panel dataset."""
        X, y = _create_panel_dataset(n_timepoints=50, n_series=2)

        # Clone the effect to ensure fresh state
        effect = effect_instance.clone()

        # Create model with the effect
        model = Prophetverse(
            trend="flat",
            exogenous_effects=[
                ("channel1", effect, "channel1"),
            ],
            inference_engine=PriorPredictiveInferenceEngine(num_samples=5),
        )

        # Fit the model
        model.fit(X=X, y=y)

        # Create budget optimizer
        budget_optimizer = BudgetOptimizer(
            objective=MaximizeKPI(),
            constraints=[TotalBudgetConstraint()],
            options={"maxiter": 10},  # Small number of iterations for testing
        )

        # Define optimization horizon
        dates = X.index.get_level_values(-1).unique()
        horizon = dates[-10:]

        # Run optimization
        X_opt = budget_optimizer.optimize(
            model=model,
            X=X,
            horizon=horizon,
            columns=["channel1"],
        )

        # Assertions
        assert X_opt is not None
        assert X_opt.shape == X.shape
        assert X_opt.index.equals(X.index)
        assert X_opt.columns.equals(X.columns)

        # Check that gradients are non-zero (budget optimization requires gradients)
        jac_values = budget_optimizer.jac_(budget_optimizer.x0_, budget_optimizer)
        assert jnp.any(
            jnp.abs(jac_values) > 1e-10
        ), f"Gradients are all zero for {type(effect_instance).__name__}"

    @pytest.mark.parametrize("effect_instance", ALL_EFFECT_TEST_CASES)
    def test_gradient_flow_through_effect(self, effect_instance):
        """Test that gradients flow correctly through the effect.

        This tests the core requirement for budget optimization: that the
        effect does not break the gradient flow from input to output.
        """
        import jax

        X, y = _create_simple_dataset(n_timepoints=30)

        # Clone the effect to ensure fresh state
        effect = effect_instance.clone()

        # Fit the effect
        effect.fit(y=y, X=X, scale=1.0)

        # Get forecast horizon
        fh = X.index[10:]

        # Transform the data
        applies_to = effect.get_tag("applies_to", tag_value_default="X")
        if applies_to == "X":
            data = effect.transform(X, fh=fh)
        else:
            data = effect.transform(y, fh=fh)

        if data is None:
            pytest.skip(
                f"Effect {type(effect_instance).__name__} returns None from transform"
            )

        # Check if effect has _update_data method (required for budget optimization)
        if not hasattr(effect, "_update_data"):
            pytest.skip(
                f"Effect {type(effect_instance).__name__} "
                "does not have _update_data method"
            )

        # Test gradient flow through _update_data
        raw_data = jnp.array(X.values, dtype=jnp.float32)

        def scalar_output_fn(x_input):
            updated_data = effect._update_data(data, x_input)
            leaves = jax.tree_util.tree_leaves(updated_data)
            total = 0.0
            for leaf in leaves:
                if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                    total = total + jnp.sum(leaf)
            return total

        # Calculate gradient
        grads = jax.grad(scalar_output_fn)(raw_data)

        # Assertions
        assert grads.shape == raw_data.shape
        assert jnp.all(
            jnp.isfinite(grads)
        ), f"Gradients contain NaNs or Infs for {type(effect_instance).__name__}"
        # At least some gradients should be non-zero
        assert jnp.any(grads != 0), (
            f"All gradients are zero for {type(effect_instance).__name__}. "
            "This means _update_data might be breaking the gradient flow."
        )
