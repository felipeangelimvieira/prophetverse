# constraints_optuna.py

import numpy as np
from prophetverse.experimental.budget_optimization.base import BaseConstraint


class SharedBudgetConstraint(BaseConstraint):
    _tags = {
        "name": "ShareBudget",
        "backend": "optuna",
    }

    def __init__(self, channels, total):
        self.channels = channels
        self.total = total
        super().__init__()

    def __call__(self, trial, X, horizon, columns):
        # Sample a “proportion” for each channel, then renormalize so they sum to `total`
        props = [trial.suggest_float(f"prop_{ch}", 0.0, 1.0) for ch in self.channels]
        s = sum(props)
        if s <= 0:
            # fallback to equal split if sampler gives all zeros
            budgets = [self.total / len(self.channels)] * len(self.channels)
        else:
            budgets = [p / s * self.total for p in props]

        return budgets
