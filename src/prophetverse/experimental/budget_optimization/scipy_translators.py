import numpy as np
from prophetverse.experimental.budget_optimization.registry import (
    register_constraint_translator,
    register_utility_translator,
)
from prophetverse.experimental.budget_optimization.translation_result import (
    TranslationResult,
)
from prophetverse.experimental.budget_optimization.constraints.base import (
    Constraint,
    UtilityFunction,
)

# Example built-in Constraint subclasses
from prophetverse.experimental.budget_optimization.constraints.builtins import (
    ShareBudget,
    ChannelCap,
)

# Example built-in UtilityFunction subclass
from prophetverse.experimental.budget_optimization.constraints.builtins import MaxROI


@register_constraint_translator("scipy", ShareBudget)
def scipy_share_budget(c: ShareBudget, model) -> TranslationResult:
    # equality: sum(x) == c.total
    n = len(c.channels) * len(c.window)
    A_eq = np.ones((1, n))
    b_eq = np.array([c.total])
    return TranslationResult(kwargs={"A_eq": A_eq, "b_eq": b_eq})


@register_constraint_translator("scipy", ChannelCap)
def scipy_channel_cap(c: ChannelCap, model) -> TranslationResult:
    # bounds: 0 <= x_t <= c.max_per_period
    n = len(c.months)
    bounds = [(0, c.max_per_period)] * n
    return TranslationResult(kwargs={"bounds": bounds})


@register_utility_translator("scipy", UtilityFunction)
# Note: ideally, register specifically for MaxROI
def scipy_max_roi(u: MaxROI, model) -> TranslationResult:
    def objective(x_array):
        # reshape & predict
        df_new = model.X_orig.copy()
        df_new[u.channels] = x_array.reshape(-1, len(u.channels))
        preds = model.predict(df_new)
        roi = preds[u.channels].sum(axis=0) / x_array
        return -float(roi.sum())

    return TranslationResult(kwargs={}, objective_fn=objective)
