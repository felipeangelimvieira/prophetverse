import jax.numpy as jnp
import pytest

from prophetverse.effects.target.univariate import _build_positive_smooth_clipper


@pytest.mark.parametrize("x", [1e10, 1e3, 1, -1, -1e3, -1e10])
def test__to_positive(x):
    _to_positive = _build_positive_smooth_clipper(1e-5)
    x_positive = _to_positive(x)

    assert jnp.all(x_positive > 0)
