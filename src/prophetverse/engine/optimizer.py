"""
Optimizers module for NumPyro models.

This module defines optimizer classes for use with NumPyro models,
including a base optimizer class and specific implementations such as
AdamOptimizer and CosineScheduleAdamOptimizer.
"""

from typing import Optional

import numpyro
import optax
from numpyro.optim import _NumPyroOptim, optax_to_numpyro
from skbase.base import BaseObject

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

    _tags = {"object_type": "optimizer"}

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
