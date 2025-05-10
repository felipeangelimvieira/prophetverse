# utility_optuna.py

from prophetverse.experimental.budget_optimization.base import BaseOptimizationObjective


class MaxROIUtility(BaseOptimizationObjective):
    _tags = {
        "name": "MaxROI",
        "backend": "optuna",
    }

    def __init__(self, channels):
        self.channels = channels
        super().__init__()

    def __call__(self, model, X, horizon, columns):
        # forecasting horizon (all unique future periods)
        fh = X.index.get_level_values(-1).unique()

        def objective(x_array):
            # x_array is a flat array of len(columns)
            # reshape & inject into X
            Xt = X.copy()
            Xt.loc[horizon, columns] = x_array.reshape(-1, len(columns))

            # get channel‐level component forecasts
            preds = model.predict_components(X=Xt, fh=fh)

            # compute sum of component K P I / budget
            total_spend = x_array.sum()
            roi = preds[self.channels].sum(axis=0) / total_spend

            # return positive ROI (Optuna will maximize)
            return float(roi.sum())

        return objective


class MaximizeKPI(BaseOptimizationObjective):
    _tags = {
        "name": "MaximizeKPI",
        "backend": "optuna",
    }

    def __init__(self, scale=1e-7):
        self.scale = scale
        super().__init__()

    def __call__(self, model, X, horizon, columns):
        # forecasting horizon (all unique future periods)
        fh = X.index.get_level_values(-1).unique()

        def objective(x_array):
            # x_array is a flat array of len(columns)
            # reshape & inject into X
            Xt = X.copy()
            Xt.loc[horizon, columns] = x_array.reshape(-1, len(columns))

            # get channel‐level component forecasts
            preds = model.predict(X=Xt, fh=fh)
            preds = preds.loc[horizon]
            # return negative K P I (Optuna will maximize)
            val = -preds.values.sum() * self.scale

            return val

        return objective
