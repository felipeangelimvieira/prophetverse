"""Base classes for budget optimization"""

from skbase.base import BaseObject
import pandas as pd
import jax.numpy as jnp


class BaseOptimizationObjective(BaseObject):
    """
    Defines the optimization objective function.

    The __call__ method should be implemented to return the objective function
    that will be optimized.
    """

    _tags = {"backend": None, "name": None}

    def _objective(self, x: jnp.ndarray, budget_optimizer):

        raise NotImplementedError(
            "Utility function must be callable with model, X, period and columns"
        )

    def __call__(self, model, X, horizon, columns):
        """
        Get optimization objective function


        Parameters
        ----------
        model : Prophetverse
            Prophetverse model
        X : pd.DataFrame
            Input data
        horizon : pd.Index
            Forecast horizon
        columns : list
            List of columns to optimize

        Returns
        -------
        objective : callable
            Objective function
        """

        return self._objective


class BaseConstraint(BaseObject):
    """
    Base constraint class.
    Defines the constraint function that will be used in the optimization.
    The __call__ method should be implemented to return a dictionary
    that will be passed to scipy.minimize contraint argument.
    """

    _tags = {"backend": None, "name": None}

    def __call__(self, model, X, horizon, columns):
        """
        Callable constraint function.
        It is expected to be overridden in subclasses.
        Parameters
        ---------
        model : Prophetverse
            Prophetverse model
        X : pd.DataFrame
            Input data
        horizon : pd.Index
            Forecast horizon
        columns : list
            List of columns to optimize

        Returns
        -------
        constraint : dict
            Dictionary with the constraint function and its jacobian, and type.
        """

        raise NotImplementedError(
            "Constraint function must be callable with model, X, period and columns"
        )


class BaseParametrizationTransformation(BaseObject):
    """
    Decision variable transformation class.

    Decision variable transforms change the parametrization of the decision
    variable. The default parametrization is a flatten array with the inputs
    for all the columns. The transform is used to change the initial guess
    passed to scipy.minimize and to inverse_transform this guess to the original
    space so that the constraints and objective function can be evaluated.

    """

    def fit(self, X, horizon, columns):
        """Fit the decision variable to the data"""
        self.horizon_ = horizon
        self.columns_ = columns
        return self._fit(X, horizon, columns)

    def transform(self, x):
        """Transform the decision variable to the original space"""
        return self._transform(x)

    def inverse_transform(self, xt):
        """Return the initial guess for the decision variable"""
        return self._inverse_transform(xt)

    def _fit(self, X: pd.DataFrame, horizon: pd.Index, columns: pd.Index):
        """Default private fit"""
        pass

    def _transform(self, x: jnp.ndarray):
        """Transform the decision variable to the original space"""
        raise NotImplementedError("Decision variable must implement transform method")

    def _inverse_transform(self, xt: jnp.ndarray):
        """Return the initial guess for the decision variable"""
        raise NotImplementedError(
            "Decision variable must implement initial_guess method"
        )


class BaseBudgetOptimizer(BaseObject):
    """Base class for budget optimization.

    Budget optimization is an optimization of a set of input variables.
    Optimizers are meant to be used in conjunction with a model. The arguments
    should be divided as follows:

    * init arguments:
        * utility function : an specification of the objective function to be
            optimized
        * contraints: a list of constraints, such as Budget, ChannelCap, etc.
        * optimization related arguments
    * optimize arguments:
        * model
        * X : pd.DataFrame
        * period : pd.PeriodIndex
        * columns : list[str]
    """

    _tags = {"backend": None}

    def __init__(
        self, constraints: list[BaseConstraint], objective: BaseOptimizationObjective
    ):
        self.constraints = constraints
        self.objective = objective
        super().__init__()

    def optimize(self, model, X: pd.DataFrame, horizon: pd.Index, columns: pd.Index):
        return self._optimize(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )
