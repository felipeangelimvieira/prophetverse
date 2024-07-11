"""Configure tests and declare global fixtures."""

import warnings

import jax.numpy as jnp
import numpyro
import pandas as pd
import pytest

from prophetverse.effects.base import BaseEffect

warnings.filterwarnings("ignore")


def pytest_sessionstart(session):
    """Avoid NaNs in tests."""
    numpyro.enable_x64()


@pytest.fixture(name="effects_sample_data")
def sample_data():
    """Sample data used at effects tests."""
    return pd.DataFrame(
        {
            "x1": range(10),
            "x2": range(10, 20),
            "log_x1": [0.1 * i for i in range(10)],
            "lin_x2": [0.2 * i for i in range(10, 20)],
        }
    )


class ConcreteEffect(BaseEffect):
    """Most simple class to test abstracteffect methods."""

    _tags = {"skip_apply_if_no_match": False}

    def __init__(self, id: str, regex=None, effect_mode="additive"):
        super().__init__(id=id, regex=regex, effect_mode=effect_mode)

    def _apply(self, trend: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Calculate simple effect."""
        return jnp.mean(data, axis=0)


@pytest.fixture(name="effect_with_regex")
def effect_with_regex():
    """Most simple class of abstracteffect with optional regex."""
    return ConcreteEffect(id="test_effect", regex="x1|x2")


@pytest.fixture
def effect_without_regex():
    """Most simple class of abstracteffect without optional regex."""
    return ConcreteEffect(id="test_effect")
