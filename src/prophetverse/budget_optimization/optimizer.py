import scipy.optimize
from prophetverse.budget_optimization.base import (
    BaseBudgetOptimizer,
    BaseConstraint,
    BaseOptimizationObjective,
    BaseParametrizationTransformation,
)
import scipy
import warnings
import numpy as np
from jax import grad, hessian
import jax.numpy as jnp
import jax
import pandas as pd
from prophetverse.budget_optimization.parametrization_transformations import (
    IdentityTransform,
)
from typing import List, Optional
import numpyro

__all__ = ["BudgetOptimizer"]


class BudgetOptimizer(BaseBudgetOptimizer):
    """Budget optimizer using scipy.optimize.minimize.

    Parameters
    ----------
    objective : BaseOptimizationObjective
        Objective function object
    constraints : list of BaseConstraint
        List of constraint objects
    decision_variable_transform : BaseDecisionVariableTransform, optional
        Decision variable transform object
    method : str, optional
        Optimization method to use. Default is "SLSQP".
    tol : float, optional
        Tolerance for termination. Default is None.
    bounds : Union[List[tuple], dict[str, tuple]], optional
        Bounds for decision variables. If a list, the value is used directly
        in scipy.optimize.minimize. If a dict, the keys are the column names and
        the values are the bounds for each column. Default is (0, np.inf) for
        each column.
    options : dict, optional
        Options for the optimization method. Default is None.
    callback : callable, optional
        Callback function to be called after each iteration. Default is None.
    """

    def __init__(
        self,
        objective: BaseOptimizationObjective,
        constraints: List[BaseConstraint],
        parametrization_transform: Optional[BaseParametrizationTransformation] = None,
        method="SLSQP",
        tol=None,
        bounds=None,
        options=None,
        callback=None,
    ):

        self.method = method
        self.options = options
        self.bounds = bounds
        self.tol = tol
        self.parametrization_transform = parametrization_transform
        self.callback = callback
        super().__init__(constraints=constraints, objective=objective)

    def _optimize(self, model, X, horizon, columns):

        # To avoid errors with Scipy optimize
        numpyro.enable_x64()
        self.set_predictive_attr(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

        # Decision variable transform
        self._parametrization_transform = self.parametrization_transform
        if self.parametrization_transform is None:
            self._parametrization_transform = IdentityTransform()

        # Prepare bounds
        self._bounds = self.bounds
        if self._bounds is None:
            self._bounds = {col: (0, np.inf) for col in columns}

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

        x0 = X.loc[X.index.get_level_values(-1).isin(horizon), columns].values.flatten()
        x0 = x0.astype(jnp.float64)
        # Transform decision variable
        self._parametrization_transform.fit(X, horizon, columns)
        x0 = self._parametrization_transform.transform(x0)

        self.x0_ = x0

        # Bounds are set based on the re-parametrized decision variable.
        self.bounds_ = []
        if isinstance(self._bounds, list):
            self.bounds_ = self._bounds
        elif len(x0) % len(columns) == 0:
            size_per_column = len(x0) // len(columns)
            for col in columns:
                self.bounds_.extend(
                    [self._bounds.get(col, (0, np.inf))] * size_per_column
                )
        else:
            if self._bounds is not None:
                warnings.warn(
                    "Bounds are not set correctly. Using default bounds (0, np.inf) for each decision variable."
                )
            self.bounds_ = [(0, np.inf)] * len(x0)

        self.objective_fun_ = self.wrap_func_with_inv_transform(self.objective_fun_)
        self.jac_ = grad(self.objective_fun_)
        self.hess_ = hessian(self.objective_fun_)

        for i in range(len(self.constraints_)):
            fun = self.constraints_[i]["fun"]
            self.constraints_[i]["fun"] = self.wrap_func_with_inv_transform(fun)
            self.constraints_[i]["jac"] = grad(self.constraints_[i]["fun"])
            self.constraints_[i]["args"] = (self,)

        minimize_kwargs = dict(
            fun=self.objective_fun_,
            x0=x0,
            jac=self.jac_,
            method=self.method,
            constraints=self.constraints_,
            hess=self.hess_,
            bounds=self.bounds_,
            options=self.options,
            tol=self.tol,
            args=self,
            callback=self.callback,
        )

        if self.method in ["BFGS"]:
            # Avoid warnings about bounds and constraints
            # but warn if the user tries to use method that do not use them.
            if self.bounds is not None:
                warnings.warn(
                    f"{self.method} method does not support bounds. Ignoring bounds."
                )
            if self.constraints_:
                warnings.warn(
                    f"{self.method} method does not support constraints. Ignoring constraints."
                )

            minimize_kwargs.pop("constraints")
            minimize_kwargs.pop("bounds")
            minimize_kwargs.pop("hess")

        res = scipy.optimize.minimize(**minimize_kwargs)
        self.result_ = res

        X_opt = X.copy()
        x_opt = self._parametrization_transform.inverse_transform(res.x)
        mask = X_opt.index.get_level_values(-1).isin(horizon)
        X_opt.loc[mask, columns] = x_opt.reshape(-1, len(columns))
        return X_opt

    def set_predictive_attr(self, model, X, horizon, columns):

        fh: pd.Index = X.index.get_level_values(-1).unique()
        # Get the indexes of `horizon` in fh
        horizon_idx = jnp.array([fh.get_loc(h) for h in horizon])

        predictive = model.optimize_predictive_callable(
            X=X, horizon=horizon, columns=columns
        )
        self.predictive_no_jit_ = predictive
        self.predictive_ = jax.jit(predictive)
        self.horizon_idx_ = horizon_idx

    def wrap_func_with_inv_transform(self, fun):
        """Wrap a function with parametrization inverse transform"""

        def wrapper(x, *args):
            xt = self._parametrization_transform.inverse_transform(x)
            return fun(xt, *args)

        return wrapper

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from prophetverse.budget_optimization.objectives import (
            MaximizeROI,
            MaximizeKPI,
            MinimizeBudget,
        )
        from prophetverse.budget_optimization.constraints import (
            TotalBudgetConstraint,
            MinimumTargetResponse,
        )
        from prophetverse.budget_optimization.parametrization_transformations import (
            InvestmentPerChannelTransform,
            TotalInvestmentTransform,
            InvestmentPerChannelAndSeries,
            InvestmentPerSeries,
        )

        params = []

        for parametrization in [
            None,
            InvestmentPerChannelTransform(),
            TotalInvestmentTransform(),
            InvestmentPerChannelAndSeries(),
            InvestmentPerSeries(),
        ]:

            if not isinstance(
                parametrization, (TotalInvestmentTransform, InvestmentPerSeries)
            ):
                params.extend(
                    [
                        {
                            "objective": MaximizeKPI(),
                            "constraints": [
                                TotalBudgetConstraint(),
                            ],
                            "parametrization_transform": parametrization,
                            "options": {"maxiter": 10},
                        },
                        {
                            "objective": MaximizeROI(),
                            "constraints": [
                                TotalBudgetConstraint(),
                                MinimumTargetResponse(0.5),
                            ],
                            "options": {"maxiter": 10},
                        },
                    ]
                )

            params.extend(
                [
                    {
                        "objective": MinimizeBudget(),
                        "constraints": [
                            MinimumTargetResponse(0.5),
                        ],
                        "options": {"maxiter": 10},
                    },
                ]
            )
        return params
