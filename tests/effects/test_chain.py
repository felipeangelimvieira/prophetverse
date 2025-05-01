"""Pytest for Chained Effects class."""

import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects.base import BaseEffect
from prophetverse.effects.chain import ChainedEffects


class MockEffect(BaseEffect):
    def __init__(self, value):
        self.value = value
        super().__init__()

        self._transform_called = False

    def _transform(self, X, fh):
        self._transform_called = True
        return super()._transform(X, fh)

    def _predict(self, data, predicted_effects, params):
        param = self.sample("param", numpyro.distributions.Delta(self.value))
        return data * param


@pytest.fixture
def index():
    return pd.date_range("2021-01-01", periods=6)


@pytest.fixture
def y(index):
    return pd.DataFrame(index=index, data=[1] * len(index))


@pytest.fixture
def X(index):
    return pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60]},
        index=index,
    )


def test_chained_effects_fit(X, y):
    """Test the fit method of ChainedEffects."""
    effects = [MockEffect(2), MockEffect(3)]
    chained = ChainedEffects(steps=effects)

    scale = 1
    chained.fit(y=y, X=X, scale=scale)
    # Ensure no exceptions occur in fit


def test_chained_effects_transform(X, y):
    """Test the transform method of ChainedEffects."""
    effects = [MockEffect(2), MockEffect(3)]
    chained = ChainedEffects(steps=effects)
    transformed = chained.transform(X, fh=X.index)
    expected = MockEffect(2).transform(X, fh=X.index)
    assert jnp.allclose(transformed, expected), "Chained transform output mismatch."


def test_chained_effects_predict(X, y):
    """Test the predict method of ChainedEffects."""
    effects = [MockEffect(2), MockEffect(3)]
    chained = ChainedEffects(steps=effects)
    chained.fit(y=y, X=X, scale=1)
    data = chained.transform(X, fh=X.index)
    predicted_effects = {}

    with numpyro.handlers.trace() as exec_trace:
        predicted = chained.predict(data, predicted_effects)
    expected = data * 2 * 3
    assert jnp.allclose(predicted, expected), "Chained predict output mismatch."


def test_get_params():
    effects = [MockEffect(2), MockEffect(3)]
    chained = ChainedEffects(steps=effects)

    params = chained.get_params()

    assert params["effect_0__value"] == 2, "Incorrect effect_0 param."
    assert params["effect_1__value"] == 3, "Incorrect effect_1 param."
