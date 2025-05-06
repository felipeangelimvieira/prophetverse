from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from prophetverse.experimental.budget_optimization.constraints.base import (
    Constraint,
    UtilityFunction,
)


@dataclass(frozen=True)
class ShareBudget(Constraint):
    channels: Sequence[str]
    total: float
    window: pd.PeriodIndex


@dataclass(frozen=True)
class ChannelCap(Constraint):
    months: pd.PeriodIndex
    max_per_period: float


@dataclass(frozen=True)
class MaxROI(UtilityFunction):
    channels: Sequence[str]
