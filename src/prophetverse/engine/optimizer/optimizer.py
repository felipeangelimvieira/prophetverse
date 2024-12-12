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
    """

    def __init__(self, step_size=0.001):
        """
        Initialize the AdamOptimizer.

        Parameters
        ----------
        step_size : float, optional
            The step size (learning rate) for the optimizer. Default is 0.001.
        """
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
    """

    def __init__(self, init_value=0.001, decay_steps=100_000, alpha=0.0, exponent=1):
        """
        Initialize the CosineScheduleAdamOptimizer.

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


class BFGSOptimizer(BaseOptimizer):

    def __init__(self, learning_rate=1e-3, memory_size=10, scale_init_precond=True):

        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.scale_init_precond = scale_init_precond
        super().__init__()

    def create_optimizer(self):

        # Linesearch
        linesearch = optax.scale_by_lbfgs(
            memory_size=self.memory_size,
            scale_init_precond=self.scale_init_precond,
        )

        # Optimizer
        opt = optax.chain(linesearch, optax.scale(-1.0))

        return numpyro.optim.optax_to_numpyro(opt)


class LBFGSSolver(BaseOptimizer):

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

        new_obj = self.clone()
        new_obj.max_iter = max_iter
        return new_obj

    def create_optimizer(self):
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


class _LegacyNumpyroOptimizer(BaseOptimizer):
    """
    Legacy optimizer.

    This class allows the use of any optimizer available in numpyro.optim by name.
    """

    def __init__(
        self, optimizer_name: str = "Adam", optimizer_kwargs: Optional[dict] = None
    ):
        """
        Initialize the _LegacyNumpyroOptimizer.

        Parameters
        ----------
        optimizer_name : str, optional
            The name of the optimizer to use from numpyro.optim. Default is "Adam".
        optimizer_kwargs : dict, optional
            Keyword arguments to pass to the optimizer. Default is None.
        """
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        super().__init__()

        self._optimizer_kwargs = optimizer_kwargs
        if self.optimizer_name == "Adam" and optimizer_kwargs is None:
            self._optimizer_kwargs = {"step_size": 0.001}
        elif self.optimizer_name != "Adam" and optimizer_kwargs is None:
            self._optimizer_kwargs = {}

    def create_optimizer(self) -> _NumPyroOptim:
        """
        Create and return a NumPyro optimizer instance.

        Returns
        -------
        _NumPyroOptim
            An instance of the specified NumPyro optimizer.

        Raises
        ------
        AttributeError
            If the specified optimizer_name is not found in numpyro.optim.
        """
        return getattr(numpyro.optim, self.optimizer_name)(**self._optimizer_kwargs)


class _OptimizerFromCallable(BaseOptimizer):
    """Temporary class to support legacy optimizer factories."""

    def __init__(self, func):

        self.func = func
        super().__init__()

    def create_optimizer(self) -> _NumPyroOptim:
        return self.func()
