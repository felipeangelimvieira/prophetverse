"""Constraints for Budget Optimizer"""

import numpy as np
from prophetverse.budget_optimization.base import (
    BaseConstraint,
)
import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad

__all__ = [
    "TotalBudgetConstraint",
]


class TotalBudgetConstraint(BaseConstraint):
    """Shared budget constraint.

    This constraint ensures that the sum of the budgets for the specified
    channels is equal to the total budget.

    Parameters
    ----------
    channels : list, optional
        List of channels to be constrained. If None, all channels are used.
    total : float, optional
        Total budget. If None, the total budget is computed from the input data.
    """

    def __init__(self, channels=None, total=None):
        self.channels = channels
        self.total = total
        self.constraint_type = "eq"
        super().__init__()

    def __call__(self, X: pd.DataFrame, horizon: pd.Index, columns: list):
        """
        Return optimization constraint definition.
        """

        channels = self.channels
        if channels is None:
            channels = columns

        total = self.total
        if total is None:
            mask = X.index.get_level_values(-1).isin(horizon)
            total = X.loc[mask, columns].sum(axis=0).sum()

        channel_idx = [columns.index(ch) for ch in channels]

        def func(x_array: jnp.ndarray, *args):
            """
            Return >=0 if the sum of the budgets for the specified channels.
            """
            x_array = x_array.reshape(-1, len(channels))
            channels_budget = x_array[:, channel_idx]
            val = total - np.sum(channels_budget)
            return val

        return {"type": self.constraint_type, "fun": func, "jac": grad(func)}


# TODO: remove in future version
SharedBudgetConstraint = TotalBudgetConstraint


class MinimumTargetResponse(BaseConstraint):
    """Minimum target response constraint.

    This constraint ensures that the target response is greater than or equal
    to a specified value. This imposes a restriction on the **output** of the
    model, instead of the input.

    Parameters
    ----------
    target_response : float
        Target response value. The model output must be greater than or equal
        to this value.
    """

    def __init__(self, target_response: float, constraint_type="ineq"):
        self.target_response = target_response
        self.constraint_type = constraint_type
        super().__init__()

    def __call__(self, X, horizon, columns):

        fh: pd.Index = X.index.get_level_values(-1).unique()
        X = X.copy()

        # Get the indexes of `horizon` in fh
        horizon_idx = jnp.array([fh.get_loc(h) for h in horizon])

        def func(x_array, budget_optimizer, *args):
            """
            Return >=0 if the target response is greater than or equal to the
            specified value.
            """
            obs = budget_optimizer.predictive_(x_array)
            out = obs.mean(axis=0).squeeze(-1)
            out = out[..., budget_optimizer.horizon_idx_].sum()
            out = out - self.target_response

            return out

        return {
            "type": self.constraint_type,
            "fun": func,
            "jac": grad(func),
        }
