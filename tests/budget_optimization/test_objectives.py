import jax.numpy as jnp
import pytest
from prophetverse.budget_optimization.objectives import (
    MaximizeROI,
    MaximizeKPI,
    MinimizeBudget,
)


class DummyOptimizer:
    def __init__(self, predictive_array, horizon_idx):
        # enforce ndarray types
        self._arr = jnp.array(predictive_array)
        self.horizon_idx_ = jnp.array(horizon_idx)

    def predictive_(self, x):
        return self._arr


def test_maximize_roi_basic():
    x = jnp.array([1.0, 2.0])
    # predictive returns two samples over three horizons
    pred = jnp.array(
        [
            [[1.0], [2.0], [3.0]],
            [[1.0], [2.0], [3.0]],
        ]
    )
    optimizer = DummyOptimizer(pred, jnp.array([1, 2]))
    obj = MaximizeROI()
    # mean over samples => [1,2,3]; pick idx [1,2] => [2,3], sum=5; spend=1+2=3 => -5/3
    result = float(obj._objective(x, optimizer))
    assert result == pytest.approx(-5.0 / 3.0)


def test_maximize_kpi_basic():
    x = jnp.array([0.5, 1.5])
    # two samples over two horizons
    pred = jnp.array(
        [
            [[2.0], [4.0]],
            [[2.0], [4.0]],
        ]
    )
    optimizer = DummyOptimizer(pred, jnp.array([0, 1]))
    obj = MaximizeKPI()
    # mean=>[2,4]; pick [0,1]=>[2,4]; sum(axis=0) over last dim yields [2,4], then sum()=6 => -6
    result = float(obj._objective(x, optimizer))
    assert result == pytest.approx(-6.0)


def test_minimize_budget_basic():
    x = jnp.array([1.0, 1.0, 3.0])

    pred = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    optimizer = DummyOptimizer(pred, jnp.array([1, 2]))
    obj = MinimizeBudget()
    # sum(x)=5
    result = float(obj._objective(x, optimizer))
    assert result == pytest.approx(5.0)
