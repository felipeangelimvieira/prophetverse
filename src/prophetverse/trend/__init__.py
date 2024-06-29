"""
Trend models to be used in ProphetVerse.

This module contains the following trend models:

- FlatTrend: A flat trend model that does not change over time.
- PiecewiseLinearTrend: A piecewise linear trend model that changes linearly over time.
- PiecewiseLogisticTrend: A piecewise logistic trend model that changes logistically
    over time.
"""

from .base import TrendModel
from .flat import FlatTrend
from .piecewise import PiecewiseLinearTrend, PiecewiseLogisticTrend

__all__ = [
    "TrendModel",
    "FlatTrend",
    "PiecewiseLinearTrend",
    "PiecewiseLogisticTrend",
]
