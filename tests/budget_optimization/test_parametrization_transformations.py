import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest

from prophetverse.experimental.budget_optimization.parametrization_transformations import (
    IdentityTransform,
    InvestmentPerChannelTransform,
)


def test_identitytransform_scalar_and_array():
    t = IdentityTransform()
    # scalar
    assert t.transform(42) == 42
    assert t.inverse_transform(42) == 42
    # array
    arr = jnp.array([1.0, 2.0, 3.0])
    out = t.transform(arr)
    assert isinstance(out, jnp.ndarray)
    assert jnp.array_equal(out, arr)
    assert jnp.array_equal(t.inverse_transform(arr), arr)


def test_investment_per_channel_roundtrip():
    # 2 days, 2 channels
    X = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], index=[0, 1], columns=["a", "b"])
    t = InvestmentPerChannelTransform()
    t.fit(X, horizon=[0, 1], columns=["a", "b"])
    x0 = X.values.flatten()
    xt = t.transform(x0)
    x_rec = t.inverse_transform(xt)
    assert x_rec.shape == x0.shape
    assert np.allclose(x_rec, x0)


def test_investment_per_channel_transform_sums():
    X = pd.DataFrame([[5.0, 7.0], [9.0, 11.0]], index=["d1", "d2"], columns=["a", "b"])
    t = InvestmentPerChannelTransform()
    t.fit(X, horizon=["d1", "d2"], columns=["a", "b"])
    x_flat = X.values.flatten()
    sums = t.transform(x_flat)
    assert sums.shape == (2,)
    assert np.allclose(sums, np.array([14.0, 18.0]))


def test_investment_per_channel_inverse_scaling():
    # daily shares: col a -> [1/2, 1/2], col b -> [9/10, 1/10]
    X = pd.DataFrame([[1.0, 9.0], [1.0, 1.0]], index=["d1", "d2"], columns=["a", "b"])
    t = InvestmentPerChannelTransform()
    t.fit(X, horizon=["d1", "d2"], columns=["a", "b"])
    xt = np.array([2.0, 4.0])
    result = t.inverse_transform(xt)
    expected = np.array([1.0, 3.6, 1.0, 0.4])  # Day1: [1, 3.6], Day2: [1, 0.4]
    assert result.shape == (4,)
    assert np.allclose(result, expected)
