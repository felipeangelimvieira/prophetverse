import jax.numpy as jnp
import pytest

from prophetverse.models import _to_positive


@pytest.mark.parametrize("x", [1e10, 1e3, 1, -1, -1e3, -1e10])
def test__to_positive(x):
    x_positive = _to_positive(x, 1e-5)

    assert jnp.all(x_positive > 0)
