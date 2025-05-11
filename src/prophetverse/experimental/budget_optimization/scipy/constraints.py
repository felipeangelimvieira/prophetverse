import numpy as np
from prophetverse.experimental.budget_optimization.base import (
    BaseConstraint,
)
import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad

__all__ = [
    "SharedBudgetConstraint",
]


class SharedBudgetConstraint(BaseConstraint):
    _tags = {
        "name": "ShareBudget",
        "backend": "scipy",
    }

    def __init__(self, channels=None, total=None):
        self.channels = channels
        self.total = total
        self.constraint_type = "eq"
        super().__init__()

    def __call__(self, X, horizon, columns):

        channels = self.channels
        if channels is None:
            channels = columns

        total = self.total
        if total is None:
            total = X.loc[horizon, columns].sum(axis=0).sum()

        channel_idx = [columns.index(ch) for ch in channels]

        def func(x_array, *args):
            x_array = x_array.reshape(-1, len(self.channels))
            channels_budget = x_array[:, channel_idx]
            val = total - np.sum(channels_budget)
            return val

        return {"type": self.constraint_type, "fun": func, "jac": grad(func)}


class MinimumTargetResponse(BaseConstraint):
    _tags = {
        "name": "MinimumTargetResponse",
        "backend": "scipy",
    }

    def __init__(self, target_response):
        self.target_response = target_response
        super().__init__()

    def __call__(self, X, horizon, columns):

        fh: pd.Index = X.index.get_level_values(-1).unique()
        X = X.copy()

        # Get the indexes of `horizon` in fh
        horizon_idx = jnp.array([fh.get_loc(h) for h in horizon])

        def func(x_array, budget_optimizer, *args):
            obs = budget_optimizer.predictive_(x_array)
            out = obs.mean(axis=0)[horizon_idx].sum()
            return out - self.target_response

        return {
            "type": "ineq",
            "fun": func,
            "jac": grad(func),
        }
