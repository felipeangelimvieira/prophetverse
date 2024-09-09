"""Test the sktime contract for Prophet and HierarchicalProphet."""

import gc  # noqa: F401

import pytest  # noqa: F401
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from prophetverse.sktime import HierarchicalProphet, Prophetverse

PROPHET_MODELS = [Prophetverse, HierarchicalProphet]


@parametrize_with_checks(PROPHET_MODELS)
def test_sktime_api_compliance(obj, test_name):
    """Test the sktime contract for Prophet and HierarchicalProphet."""
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
