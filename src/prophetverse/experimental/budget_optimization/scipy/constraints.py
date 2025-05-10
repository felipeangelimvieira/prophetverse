import numpy as np
from prophetverse.experimental.budget_optimization.base import (
    BaseConstraint,
)

__all__ = [
    "SharedBudgetConstraint",
]


class SharedBudgetConstraint(BaseConstraint):
    _tags = {
        "name": "ShareBudget",
        "backend": "scipy",
    }

    def __init__(self, channels, total, constraint_type="eq"):
        self.channels = channels
        self.total = total
        self.constraint_type = constraint_type
        super().__init__()

    def __call__(self, X, horizon, columns):

        channel_idx = [columns.index(ch) for ch in self.channels]

        def func(x_array):
            x_array = x_array.reshape(-1, len(self.channels))
            channels_budget = x_array[:, channel_idx]
            val = self.total - np.sum(channels_budget)
            return val

        return {"type": self.constraint_type, "fun": func}
