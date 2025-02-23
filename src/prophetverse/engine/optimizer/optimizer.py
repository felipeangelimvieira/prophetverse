"""
Optimizers module for NumPyro models.

This module defines optimizer classes for use with NumPyro models,
including a base optimizer class and specific implementations such as
AdamOptimizer and CosineScheduleAdamOptimizer.
"""

from typing import Optional

import jax.numpy as jnp
import numpyro
import optax
from numpyro.optim import _NumPyroOptim, optax_to_numpyro
from skbase.base import BaseObject

from prophetverse.engine.optimizer._lbfgs import LBFGS

__all__ = [
    "BaseOptimizer",
    "AdamOptimizer",
    "CosineScheduleAdamOptimizer",
    "LBFGSSolver",
]


class BaseOptimizer(BaseObject):
    """
    Base class for optimizers in NumPyro.

    This abstract base class defines the interface that all optimizers must implement.
    """

    _tags = {"object_type": "optimizer", "is_solver": False}

    def __init__(self):
        """
        Initialize the BaseOptimizer.

        Since this is an abstract base class, initialization does not do anything.
        """
        ...

    def create_optimizer(self) -> _NumPyroOptim:
        """
        Create and return a NumPyro optimizer instance.

        Returns
        -------
        _NumPyroOptim
            An instance of a NumPyro optimizer.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        raise NotImplementedError(
            "create_optimizer method must be implemented in subclass"
        )


class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer for NumPyro models.

    This class implements the Adam optimization algorithm.

    Parameters
    ----------
    step_size : float, optional
        The step size (learning rate) for the optimizer. Default is 0.001.
    """

    def __init__(self, step_size=0.001):
        """Initialize the AdamOptimizer."""
        self.step_size = step_size
        super().__init__()

    def create_optimizer(self) -> _NumPyroOptim:
        """
        Create and return a NumPyro Adam optimizer instance.

        Returns
        -------
        _NumPyroOptim
            An instance of NumPyro's Adam optimizer.
        """
        return numpyro.optim.Adam(step_size=self.step_size)


class CosineScheduleAdamOptimizer(BaseOptimizer):
    """
    Adam optimizer with cosine decay learning rate schedule.

    This optimizer combines the Adam optimizer with a cosine decay schedule
    for the learning rate.

    Parameters
    ----------
    init_value : float, optional
        Initial learning rate. Default is 0.001.
    decay_steps : int, optional
        Number of steps over which the learning rate decays. Default is 100_000.
    alpha : float, optional
        Final multiplier for the learning rate. Default is 0.0.
    exponent : int, optional
        Exponent for the cosine decay schedule. Default is 1.
    """

    def __init__(self, init_value=0.001, decay_steps=100_000, alpha=0.0, exponent=1):
        self.init_value = init_value
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.exponent = exponent
        super().__init__()

    def create_optimizer(self) -> _NumPyroOptim:
        """
        Create and return a NumPyro optimizer with cosine decay schedule.

        Returns
        -------
        _NumPyroOptim
            An instance of a NumPyro optimizer with cosine decay schedule.
        """
        scheduler = optax.cosine_decay_schedule(
            init_value=self.init_value,
            decay_steps=self.decay_steps,
            alpha=self.alpha,
            exponent=self.exponent,
        )

        opt = optax_to_numpyro(
            optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1.0),
            )
        )
        return opt


class LBFGSSolver(BaseOptimizer):
    """
    L-BFGS solver.

    This solver is more practical than other optimizers since it usually does not
    require tuning of hyperparameters to get better estimates.

    If your model does not converge with the default hyperparameters, you can try
    increasing `memory_size`, `max_linesearch_steps`, or setting a larger number
    of steps.

    Parameters
    ----------
    gtol : float, default=1e-6
        Gradient tolerance for stopping criterion.
    tol : float, default=1e-6
        Function value tolerance for stopping criterion.
    learning_rate : float, default=1e-3
        Initial learning rate.
    memory_size : int, default=10
        Memory size for L-BFGS updates.
    scale_init_precond : bool, default=True
        Whether to scale the initial preconditioner.
    max_linesearch_steps : int, default=20
        Maximum number of line search steps.
    initial_guess_strategy : str, default="one"
        Strategy for the initial line search step size guess.
    max_learning_rate : float, optional
        Maximum allowed learning rate during line search.
    linesearch_tol : float, default=0
        Tolerance parameter for line search.
    increase_factor : float, default=2
        Factor by which to increase step size during line search when conditions are
        met.
    slope_rtol : float, default=0.0001
        Relative tolerance for slope in the line search.
    curv_rtol : float, default=0.9
        Curvature condition tolerance for line search.
    approx_dec_rtol : float, default=0.000001
        Approximate decrease tolerance for line search.
    stepsize_precision : float, default=1e5
        Stepsize precision tolerance.
    """

    _tags = {
        "is_solver": True,
    }

    def __init__(
        self,
        gtol: float = 1e-6,
        tol: float = -jnp.inf,
        learning_rate=1e-3,
        memory_size=50,
        scale_init_precond=True,
        # linesearch
        max_linesearch_steps=50,
        initial_guess_strategy="one",
        max_learning_rate=None,
        linesearch_tol=0,
        increase_factor=2,
        slope_rtol=0.0001,
        curv_rtol=0.9,
        approx_dec_rtol=0.000001,
        stepsize_precision=1e5,
    ):
        self.gtol = gtol
        self.tol = tol
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.scale_init_precond = scale_init_precond

        # Linesearch
        self.max_linesearch_steps = max_linesearch_steps
        self.initial_guess_strategy = initial_guess_strategy
        self.max_learning_rate = max_learning_rate
        self.linesearch_tol = linesearch_tol
        self.increase_factor = increase_factor
        self.slope_rtol = slope_rtol
        self.curv_rtol = curv_rtol
        self.approx_dec_rtol = approx_dec_rtol
        self.stepsize_precision = stepsize_precision

        super().__init__()

        self.max_iter = 1000

    def set_max_iter(self, max_iter: int):
        """Set the maximum number of iterations for the solver."""
        new_obj = self.clone()
        new_obj.max_iter = max_iter
        return new_obj

    def create_optimizer(self):
        """Create and return a NumPyro L-BFGS solver instance."""
        return LBFGS(
            max_iter=self.max_iter,
            tol=self.tol,
            gtol=self.gtol,
            learning_rate=self.learning_rate,
            memory_size=self.memory_size,
            scale_init_precond=self.scale_init_precond,
            # linesearch
            max_linesearch_steps=self.max_linesearch_steps,
            initial_guess_strategy="one",
            max_learning_rate=self.max_learning_rate,
            linesearch_tol=self.linesearch_tol,
            increase_factor=self.increase_factor,
            slope_rtol=self.slope_rtol,
            curv_rtol=self.curv_rtol,
            approx_dec_rtol=self.approx_dec_rtol,
            stepsize_precision=self.stepsize_precision,
        )
