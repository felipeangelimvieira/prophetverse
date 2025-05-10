# optimizer_optuna.py

import numpy as np
import optuna
from prophetverse.experimental.budget_optimization.base import BaseBudgetOptimizer


class OptunaBudgetOptimizer(BaseBudgetOptimizer):
    """Budget optimizer using an Optuna Study."""

    name = "optuna"

    def __init__(self, utility, constraints, opt_kwargs=None):
        """
        Args:
            utility: an instance of BaseUtility (e.g. MaxROIUtility(…))
            constraints: a list of BaseConstraint instances (e.g. [SharedBudgetConstraint(…)])
            opt_kwargs: passed through to `study.optimize`, e.g. {"n_trials":100}
        """
        self.opt_kwargs = opt_kwargs or {}
        super().__init__(constraints=constraints, objective=utility)

    def _optimize(self, model, X, horizon, columns):
        # 1) prepare the pure-ROI objective function
        self.utility_fun_ = self.objective(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

        # 2) set up the study to maximize ROI
        study = optuna.create_study(direction="maximize")

        # 3) define objective(trial)
        def objective(trial):
            # start from the baseline budgets at `horizon`
            x0 = X.loc[horizon, columns].values.flatten().copy()

            # let each constraint sample/reassign a slice of x0
            for constraint in self.constraints:
                # e.g. SharedBudgetConstraint returns a list in the order of constraint.channels
                new_budgets = constraint(trial, X, horizon, columns)
                for i, ch in enumerate(constraint.channels):
                    idx = columns.index(ch)
                    x0[idx] = new_budgets[i]

            # 4) evaluate ROI
            return self.utility_fun_(x0)

        # 5) run the optimization
        study.optimize(
            objective,
            **self.opt_kwargs,
        )
        return study
