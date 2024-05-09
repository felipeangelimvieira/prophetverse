"""Test the sktime contract for Prophet and HierarchicalProphet."""

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

from prophetverse.sktime import HierarchicalProphet, Prophet

PROPHET_MODELS = [Prophet, HierarchicalProphet]


def deep_equals_custom(x, y, return_msg=False, plugins=None):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal
    plugins : list, optional, default=None
        list of plugins to use for custom deep_equals
        entries must be functions with the signature:
        ``(x, y, return_msg: bool) -> return``
        where return is:
        ``None``, if the plugin does not apply, otheriwse:
        ``is_equal: bool`` if ``return_msg=False``,
        ``(is_equal: bool, msg: str)`` if return_msg=True.
        Plugins can have an additional argument ``deep_equals=None``
        by which the parent function to be called recursively is passed

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
    """
    ret = _make_ret(return_msg)

    if type(x) is not type(y):
        return ret(False, f".type, x.type = {type(x)} != y.type = {type(y)}")

    # we now know all types are the same
    # so now we compare values

    # recursion through lists, tuples and dicts
    if isinstance(x, (list, tuple)):
        return ret(*_tuple_equals(x, y, return_msg=True))
    elif isinstance(x, dict):
        return ret(*_dict_equals(x, y, return_msg=True))
    elif _is_npnan(x):
        return ret(_is_npnan(y), f"type(x)={type(x)} != type(y)={type(y)}")
    elif isclass(x):
        return ret(x == y, f".class, x={x.__name__} != y={y.__name__}")
    elif isinstance(x, jnp.ndarray):
        return ret(jnp.array_equal(x, y), ".ndarray")

    if plugins is not None:
        for plugin in plugins:
            # check if plugin has deep_equals argument
            # if so, pass this function as argument to plugin
            # this allows for recursive calls to deep_equals

            # get the signature of the plugin
            sig = signature(plugin)
            # check if deep_equals is an argument of the plugin
            if "deep_equals" in sig.parameters:
                # we need to pass in the same plugins, so we curry
                def deep_equals_curried(x, y, return_msg=False):
                    return deep_equals_custom(
                        x, y, return_msg=return_msg, plugins=plugins
                    )

                kwargs = {"deep_equals": deep_equals_curried}
            else:
                kwargs = {}

            res = plugin(x, y, return_msg=return_msg, **kwargs)

            # if plugin does not apply, res is None
            if res is not None:
                return res

    # this if covers case where != is boolean
    # some types return a vector upon !=, this is covered in the next elif
    if isinstance(x == y, bool):
        return ret(x == y, f" !=, {x} != {y}")

    # check if numpy is available
    numpy_available = _softdep_available("numpy")
    if numpy_available:
        import numpy as np

    # deal with the case where != returns a vector
    if numpy_available and np.any(x != y) or any(_coerce_list(x != y)):
        return ret(False, f" !=, {x} != {y}")

    return ret(True, "")

from skbase.utils.deep_equals import _deep_equals

_deep_equals.deep_equals_custom = deep_equals_custom

@pytest.mark.parametrize("model", PROPHET_MODELS)
def test_sktime_contract(model):
    """Test the sktime contract for Prophet and HierarchicalProphet."""
    check_estimator(model, raise_exceptions=True)
