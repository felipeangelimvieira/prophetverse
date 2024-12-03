"""Datasets for ProphetVerse."""

from .loaders import (
    load_forecastingdata,
    load_pedestrian_count,
    load_peyton_manning,
    load_tensorflow_github_stars,
    load_tourism,
)
from .synthetic import load_synthetic_squared_exogenous

__all__ = [
    "load_forecastingdata",
    "load_pedestrian_count",
    "load_peyton_manning",
    "load_tensorflow_github_stars",
    "load_tourism",
    "load_synthetic_squared_exogenous",
]
