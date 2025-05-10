import scipy.optimize
from prophetverse.experimental.budget_optimization.base import BaseBudgetOptimizer
import scipy
import numpy as np
from jax import grad, hessian
from prophetverse.experimental.budget_optimization.decision_variable_transform import (
    IdentityDecisionVariableTransform,
)


__all__ = ["ScipyBudgetOptimizer"]


class ScipyBudgetOptimizer(BaseBudgetOptimizer):
    """Budget optimizer using scipy.optimize.minimize.

    This class is meant to be used with a model that has a predict method.
    The predict method should return a pd.DataFrame with the same columns as
    the input X.
    """

    _tags = {
        "backend": "scipy",
    }

    def __init__(
        self,
        objective,
        constraints,
        decision_variable_transform=None,
        method="SLSQP",
        tol=None,
        bounds=None,
        options=None,
    ):

        self.method = method
        self.options = options
        self.bounds = bounds
        self.tol = tol
        self.decision_variable_transform = decision_variable_transform
        super().__init__(constraints=constraints, objective=objective)

    def _optimize(self, model, X, horizon, columns):

        # Decision variable transform
        self._decision_variable_transform = self.decision_variable_transform
        if self.decision_variable_transform is None:
            self._decision_variable_transform = IdentityDecisionVariableTransform()

        # Prepare bounds
        self._bounds = self.bounds
        if self._bounds is None:
            self._bounds = {col: (0, np.inf) for col in columns}

        self.bounds_ = []
        for col in columns:
            self.bounds_.extend([self._bounds.get(col, (0, np.inf))] * len(horizon))

        # Set constraints
        self.constraints_ = []
        for constraint in self.constraints:
            self.constraints_.append(
                constraint(
                    X=X,
                    horizon=horizon,
                    columns=columns,
                )
            )

        self.objective_fun_ = self.objective(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

        self.jac_ = grad(self.objective_fun_)
        self.hess_ = hessian(self.objective_fun_)

        x0 = X.loc[horizon, columns].values.flatten()
        self._decision_variable_transform.fit(X, horizon, columns)
        x0 = self._decision_variable_transform.transform(x0)

        def objective_fun_wrapper(x):
            xt = self._decision_variable_transform.inverse_transform(x)
            return self.objective_fun_(xt)

        res = scipy.optimize.minimize(
            fun=objective_fun_wrapper,
            x0=x0,
            constraints=self.constraints_,
            method=self.method,
            hess=self.hess_,
            jac=self.jac_,
            bounds=self.bounds_,
            options=self.options,
            tol=self.tol,
        )
        self.res_ = res

        X_opt = X.copy()
        x_opt = self._decision_variable_transform.inverse_transform(res.x)
        X_opt.loc[horizon, columns] = x_opt.reshape(-1, len(columns))
        return X_opt
