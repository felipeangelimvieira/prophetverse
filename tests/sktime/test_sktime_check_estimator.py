"""Test the sktime contract for Prophet and HierarchicalProphet."""

import functools
from inspect import isclass, signature
from typing import List

import pytest
from jax import numpy as jnp
from skbase.utils.deep_equals._common import _make_ret
from skbase.utils.deep_equals._deep_equals import (_coerce_list, _dict_equals,
                                                   _is_npnan,
                                                   _softdep_available,
                                                   _tuple_equals)
from sktime.utils.estimator_checks import check_estimator

from prophetverse.sktime import (HierarchicalProphet, Prophet, ProphetGamma,
                                 ProphetNegBinomial, Prophetverse)

PROPHET_MODELS = [
    Prophetverse,
    HierarchicalProphet,
]


@pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.parametrize("model", PROPHET_MODELS)
def test_check_estimator(model):
    """Test the sktime contract for Prophet and HierarchicalProphet."""

    check_estimator(model, raise_exceptions=True)
