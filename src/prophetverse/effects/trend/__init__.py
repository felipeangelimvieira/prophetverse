"""Module for trend models in prophetverse."""

from .base import TrendEffectMixin
from .flat import FlatTrend
from .piecewise import PiecewiseLinearTrend, PiecewiseFlatTrend, PiecewiseLogisticTrend

__all__ = [
    "TrendEffectMixin",
    "FlatTrend",
    "PiecewiseLinearTrend",
    "PiecewiseFlatTrend",
    "PiecewiseLogisticTrend",
]
