from prophetverse.budget_optimization.base import (
    BaseOptimizationObjective,
)
import jax.numpy as jnp

__all__ = [
    "MaximizeROI",
    "MaximizeKPI",
    "MinimizeBudget",
]


class MaximizeROI(BaseOptimizationObjective):
    """
    Maximize return on investment (ROI) objective function.
    """

    _tags = {
        "name": "MaxROI",
        "backend": "scipy",
    }

    def __init__(self):
        super().__init__()

    def _objective(self, x: jnp.ndarray, budget_optimizer):
        """
        Compute objective function value from `obs` site

        Parameters
        ----------
        obs : jnp.ndarray
            Observed values

        Returns
        -------
        float
            Objective function value
        """
        obs = budget_optimizer.predictive_(x)
        obs = obs.mean(axis=0).squeeze(-1)
        obs_horizon = obs[..., budget_optimizer.horizon_idx_]
        total_return = obs_horizon.sum()
        spend = x.sum()

        return -total_return / spend


class MaximizeKPI(BaseOptimizationObjective):
    """
    Maximize the KPI objective function.
    """

    def __init__(self):
        super().__init__()

    def _objective(self, x: jnp.ndarray, budget_optimizer):
        """
        Compute objective function value from `obs` site

        Parameters
        ----------
        obs : jnp.ndarray
            Observed values

        Returns
        -------
        float
            Objective function value
        """
        obs = budget_optimizer.predictive_(x)
        obs = obs.mean(axis=0).squeeze(-1)
        obs_horizon = obs[..., budget_optimizer.horizon_idx_]
        obs_horizon = obs_horizon.sum(axis=0)

        value = -obs_horizon.sum()
        return value


class MinimizeBudget(BaseOptimizationObjective):
    """
    Minimize budget constraint objective function.
    """

    def __init__(self, scale=1):
        self.scale = scale
        super().__init__()

    def _objective(self, x: jnp.ndarray, budget_optimizer):
        """
        Compute objective function value from `obs` site

        Parameters
        ----------
        obs : jnp.ndarray
            Observed values

        Returns
        -------
        float
            Objective function value
        """
        total_investment = x.sum() / self.scale

        return total_investment
