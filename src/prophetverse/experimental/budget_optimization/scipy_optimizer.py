import numpy as np
from scipy.optimize import minimize  # assuming scipy is installed

from prophetverse.experimental.budget_optimization.registry import (
    get_constraint_translator,
    get_utility_translator,
    list_translators,
    discover_entrypoint_plugins,
)
from prophetverse.experimental.budget_optimization.translation_result import (
    TranslationResult,
)
from prophetverse.experimental.budget_optimization.constraints.base import (
    Constraint,
    UtilityFunction,
)


class ScipyOptimizer:
    name = "scipy"

    def __init__(
        self,
        model,
        utility: UtilityFunction,
        constraints: list[Constraint],
        **opt_kwargs,
    ):
        # Optionally discover plugins at init
        discover_entrypoint_plugins()

        self.model = model
        self.utility = utility
        self.constraints = constraints
        self.opt_kwargs = opt_kwargs

        # Validation
        missing = []
        for c in constraints:
            if get_constraint_translator(self.name, c) is None:
                missing.append(type(c).__name__)
        if get_utility_translator(self.name, utility) is None:
            missing.append(type(utility).__name__)
        if missing:
            raise ValueError(
                f"Missing translators for backend '{self.name}': {missing}"
            )

    def optimize(self, X: "pd.DataFrame", interval: "pd.PeriodIndex"):
        # 1) translate constraints
        cons_kwargs = {}
        for c in self.constraints:
            tr_fn = get_constraint_translator(self.name, c)
            result: TranslationResult = tr_fn(c, model=self.model)
            cons_kwargs.update(result.kwargs)

        # 2) translate utility
        util_fn: TranslationResult = get_utility_translator(self.name, self.utility)(
            self.utility, model=self.model
        )
        obj_fn = util_fn.objective_fn
        jac_fn = util_fn.gradient_fn

        # 3) prepare initial x0
        x0 = self.model.X_orig[self.utility.channels].values.flatten()

        # 4) call scipy
        res = minimize(fun=obj_fn, x0=x0, jac=jac_fn, **cons_kwargs, **self.opt_kwargs)
        return res

    @classmethod
    def available_translators(cls):
        return list_translators(cls.name)
