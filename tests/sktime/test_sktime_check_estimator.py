"""Test the sktime contract for Prophet and HierarchicalProphet."""

import pytest
from sktime.utils.estimator_checks import check_estimator

from prophetverse.sktime import HierarchicalProphet, Prophetverse

PROPHET_MODELS = [
    Prophetverse,
    HierarchicalProphet,
]


@pytest.mark.parametrize("model", PROPHET_MODELS)
def test_check_estimator(model):
    """Test the sktime contract for Prophet and HierarchicalProphet."""

    check_estimator(model, raise_exceptions=True)
