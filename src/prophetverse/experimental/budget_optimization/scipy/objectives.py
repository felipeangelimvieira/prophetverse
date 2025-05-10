from prophetverse.experimental.budget_optimization.base import (
    BaseOptimizationObjective,
)
import jax.numpy as jnp
from jax import grad
import pandas as pd
from prophetverse.sktime import Prophetverse

__all__ = [
    "BaseScipyOptimizationObjective",
    "MaxROI",
    "MaximizeKPI",
]


class BaseScipyOptimizationObjective(BaseOptimizationObjective):

    _tags = {
        "backend": "scipy",
    }

    def _objective(self, obs: jnp.ndarray):
        """
        Compute objective function value from `obs` site

        Parameters
        ----------
        obs : jnp.ndarray
            Observed values

        Returns
        float
            Objective function value
        """
        raise NotImplementedError(
            "Objective function not implemented. Please implement the `_objective` method."
        )

    def __call__(self, model: Prophetverse, X, horizon, columns):
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

        def _objective(new_x):
            """
            Update predict data and call self._objective
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

            return self._objective(obs, horizon_idx, new_x)

        return _objective


class MaxROI(BaseScipyOptimizationObjective):
    _tags = {
        "name": "MaxROI",
        "backend": "scipy",
    }

    def __init__(self):
        super().__init__()

    def _objective(self, obs: jnp.ndarray, horizon_idx: jnp.ndarray, x: jnp.ndarray):
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
        obs = obs.mean(axis=0)
        obs_horizon = obs[horizon_idx]
        total_return = obs_horizon.sum(axis=0).sum()
        spend = x.sum(axis=0).sum()

        return -total_return / spend


class MaximizeKPI(BaseScipyOptimizationObjective):

    def __init__(self):
        super().__init__()

    def _objective(self, obs: jnp.ndarray, horizon_idx: jnp.ndarray, x: jnp.ndarray):
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
        obs = obs.mean(axis=0)
        obs_horizon = obs[horizon_idx]
        obs_horizon = obs_horizon.sum(axis=0)

        value = -obs_horizon.sum()
        return value
