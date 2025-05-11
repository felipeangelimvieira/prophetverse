import scipy.optimize
from prophetverse.experimental.budget_optimization.base import BaseBudgetOptimizer
import scipy
import warnings
import numpy as np
from jax import grad, hessian
from prophetverse.experimental.budget_optimization.decision_variable_transform import (
    IdentityDecisionVariableTransform,
)
from prophetverse.engine.optimizer._lbfgs import run_opt, LBFGS

__all__ = ["ScipyBudgetOptimizer"]


class OptaxLBFGSBudgetOptimizer(BaseBudgetOptimizer):
    """Budget optimizer using scipy.optimize.minimize.

    This class is meant to be used with a model that has a predict method.
    The predict method should return a pd.DataFrame with the same columns as
    the input X.
    """

    _tags = {
        "backend": "optax",
    }

    def __init__(
        self,
        objective,
        decision_variable_transform=None,
        max_iter=1000,
        tol=1e-5,
        gtol=1e-5,
        num_steps=10,
    ):

        self.max_iter = max_iter
        self.tol = tol
        self.gtol = gtol
        self.num_steps = num_steps
        self.decision_variable_transform = decision_variable_transform
        super().__init__(constraints=[], objective=objective)

    def _optimize(self, model, X, horizon, columns):

        # Decision variable transform
        self._decision_variable_transform = self.decision_variable_transform
        if self.decision_variable_transform is None:
            self._decision_variable_transform = IdentityDecisionVariableTransform()

        self.objective_fun_ = self.objective(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

        self.opt_ = LBFGS(
            memory_size=100, max_linesearch_steps=100, learning_rate=1e-2
        ).get_transformation()

        x0 = X.loc[horizon, columns].values.flatten()
        self._decision_variable_transform.fit(X, horizon, columns)
        x0 = self._decision_variable_transform.transform(x0)

        def wrap_func_with_inv_transform(fun):

            def wrapper(x):
                xt = self._decision_variable_transform.inverse_transform(x)
                return fun(xt), None

            return wrapper

        self.objective_fun_ = wrap_func_with_inv_transform(self.objective_fun_)

        for _ in range(self.num_steps):
            x0, _, self.final_value_, _ = run_opt(
                init_params=x0,
                fun=self.objective_fun_,
                opt=self.opt_,
                max_iter=self.max_iter,
                tol=self.tol,
                gtol=self.gtol,
            )

        self.final_params_ = x0
        X_opt = X.copy()
        x_opt = self._decision_variable_transform.inverse_transform(self.final_params_)
        X_opt.loc[horizon, columns] = x_opt.reshape(-1, len(columns))
        return X_opt
