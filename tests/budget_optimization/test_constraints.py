from prophetverse.experimental.budget_optimization.constraints import (
    SharedBudgetConstraint,
    MinimumTargetResponse,
)

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp


def test_shared_budget_constraint_default_channels():
    # two channels 'a','b' over 2 time points => total budget = 3+7+4+6 = 20
    X = pd.DataFrame(
        [[3, 7], [4, 6]],
        index=pd.Index([0, 1], name="horizon"),
        columns=pd.Index(["a", "b"]),
    )
    horizon = pd.Index([0, 1])
    columns = X.columns.tolist()
    c = SharedBudgetConstraint()
    spec = c(X, horizon, columns)
    # flattened budgets
    x = jnp.array([3, 7, 4, 6]).astype(jnp.float32)
    assert spec["type"] == "eq"
    # residual = total - sum(x) = 20 - 20 = 0
    assert spec["fun"](x) == pytest.approx(0)
    # gradient = d(20 - sum)/dx_i = -1 for each element
    grad = spec["jac"](x)
    assert np.allclose(np.array(grad), -1.0)


def test_shared_budget_constraint_custom_channels():
    # only channel 'a' => total = 3+4 = 7
    X = pd.DataFrame(
        [[3, 7], [4, 6]],
        index=pd.Index([0, 1], name="horizon"),
        columns=pd.Index(["a", "b"]),
    )
    c = SharedBudgetConstraint(channels=["a"])
    spec = c(X, pd.Index([0, 1]), X.columns.tolist())
    x = jnp.array([3, 7, 4, 6]).astype(jnp.float32)  # budgets for 'a' over 2 points
    # residual = 7 - (3+4) = 0
    assert spec["fun"](x) == pytest.approx(0)
    # gradient = -1 on each entry
    assert np.allclose(np.array(spec["jac"](x)), -1.0)


class DummyOptimizer:
    def __init__(self, out, horizon_idx=None):
        # out: array of shape (n_draws, total_horizon_len)
        self._out = out
        self.horizon_idx_ = (
            horizon_idx if horizon_idx is not None else jnp.array([0, 1, 2])
        )

    def predictive_(self, x):
        return self._out


def test_minimum_target_response_satisfied_and_unsatisfied():
    # build X with last level as horizon values [0,1,2]
    idx = pd.MultiIndex.from_product([["obs"], [0, 1, 2]], names=["obs_id", "horizon"])
    X = pd.DataFrame(np.zeros((3, 1)), index=idx, columns=["dummy"])
    horizon = [1, 2]
    # predictive_ returns ones => mean over draws = ones => sum at horizon=1+1=2
    out = np.ones((5, 3, 1))
    opt = DummyOptimizer(out, horizon_idx=jnp.array([1, 2]))
    c1 = MinimumTargetResponse(target_response=1.0)
    spec1 = c1(X, horizon, None)
    val1 = spec1["fun"](None, opt)
    assert val1 == pytest.approx(2.0 - 1.0)  # >=0

    # now target higher => unsatisfied
    c2 = MinimumTargetResponse(target_response=3.0)
    spec2 = c2(X, horizon, None)
    val2 = spec2["fun"](None, opt)
    assert val2 == pytest.approx(2.0 - 3.0)  # negative

    # gradient should be zero since predictive_ ignores x
    grad = spec1["jac"](jnp.zeros((4,)), opt)
    assert np.allclose(np.array(grad), 0.0)
