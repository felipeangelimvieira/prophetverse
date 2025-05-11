import numpy as np
from prophetverse.experimental.budget_optimization.base import (
    BaseConstraint,
)
from optax.projections import projection_l1_sphere

__all__ = [
    "SharedBudgetConstraint",
]


class SharedBudgetConstraint(BaseConstraint):
    _tags = {
        "name": "ShareBudget",
        "backend": "optax",
    }

    def __init__(self, total=None, constraint_type="eq"):
        self.total = total
        self.constraint_type = constraint_type
        super().__init__()

    def __call__(self, X, horizon, columns):

        total = self.total
        if total is None:
            total = X.loc[horizon, columns].sum(axis=0).sum()

        def func(x_array):
            return projection_l1_sphere(x_array)

        return {"type": self.constraint_type, "fun": func}
