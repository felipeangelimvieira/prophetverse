"""Test the sktime contract for Prophet and HierarchicalProphet."""

import pytest
from sktime.utils.estimator_checks import check_estimator

from prophetverse.sktime import HierarchicalProphet, Prophet

PROPHET_MODELS = [Prophet, HierarchicalProphet]


@pytest.mark.parametrize("model", PROPHET_MODELS)
def test_sktime_contract(model):
    """Test the sktime contract for Prophet and HierarchicalProphet."""
    check_estimator(model, raise_exceptions=True)
