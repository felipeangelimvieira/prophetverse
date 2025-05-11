import scipy.optimize
from prophetverse.experimental.budget_optimization.base import BaseBudgetOptimizer
import scipy
import warnings
import numpy as np
from jax import grad, hessian
import jax.numpy as jnp
import jax
import pandas as pd
from prophetverse.sktime import Prophetverse
from tqdm import tqdm
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

        self.set_utilities(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

        # Decision variable transform
        self._decision_variable_transform = self.decision_variable_transform
        if self.decision_variable_transform is None:
            self._decision_variable_transform = IdentityDecisionVariableTransform()

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

        x0 = X.loc[horizon, columns].values.flatten()
        self._decision_variable_transform.fit(X, horizon, columns)
        x0 = self._decision_variable_transform.transform(x0)

        self.bounds_ = []
        if isinstance(self._bounds, list):
            self.bounds_ = self._bounds
        else:
            size_per_column = len(x0) // len(columns)
            for col in columns:
                self.bounds_.extend(
                    [self._bounds.get(col, (0, np.inf))] * size_per_column
                )

        def wrap_func_with_inv_transform(fun):

            def wrapper(x, *args):
                xt = self._decision_variable_transform.inverse_transform(x)
                return fun(xt, *args)

            return wrapper

        self.objective_fun_ = wrap_func_with_inv_transform(self.objective_fun_)
        self.jac_ = grad(self.objective_fun_)
        self.hess_ = hessian(self.objective_fun_)

        for i in range(len(self.constraints_)):
            fun = self.constraints_[i]["fun"]
            self.constraints_[i]["fun"] = lambda x, *args: fun(
                self._decision_variable_transform.inverse_transform(x), *args
            )
            self.constraints_[i]["args"] = (self,)

        pbar = tqdm(total=None, desc="Optimizating", unit="it")

        def tqdm_callback(*args):
            pbar.update(1)

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
            callback=tqdm_callback,
        )

        if self.method in ["BFGS"]:
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
        self.res_ = res

        X_opt = X.copy()
        x_opt = self._decision_variable_transform.inverse_transform(res.x)
        X_opt.loc[horizon, columns] = x_opt.reshape(-1, len(columns))
        return X_opt

    def set_utilities(self, model, X, horizon, columns):

        fh: pd.Index = X.index.get_level_values(-1).unique()
        X = X.copy()

        predict_data = model.get_predict_data(X=X, fh=fh)
        inference_engine = model.inference_engine_

        # Get the indexes of `horizon` in fh
        horizon_idx = jnp.array([fh.get_loc(h) for h in horizon])

        # Prepare exogenous effects -
        # we need to transform them on every call to check the
        # objective function and gradient
        exogenous_effects_column_idx = []
        for effect_name, effect, effect_columns in model.exogenous_effects_:
            # If no columns are found, skip
            if effect_columns is None or len(effect_columns) == 0:
                continue

            intersection = effect_columns.intersection(columns)
            if len(intersection) == 0:
                continue

            exogenous_effects_column_idx.append(
                (
                    effect_name,
                    effect,
                    # index of effect_columns in columns
                    [columns.index(col) for col in intersection],
                )
            )

        x_array = jnp.array(X.values)

        @jax.jit
        def predictive(new_x):
            """
            Update predict data and call self._predict
            """
            new_x = new_x.reshape(-1, len(columns))
            for effect_name, effect, effect_column_idx in exogenous_effects_column_idx:
                _data = x_array[:, effect_column_idx]
                _data = _data.at[horizon_idx].set(new_x[:, effect_column_idx])
                # Update the effect data
                predict_data["data"][effect_name] = effect._update_data(
                    predict_data["data"][effect_name], _data
                )

            predictive_samples = inference_engine.predict(**predict_data)
            obs = predictive_samples["obs"]
            obs = obs * model._scale

            return obs

        self.predictive_ = predictive
        self.horizon_idx_ = horizon_idx
