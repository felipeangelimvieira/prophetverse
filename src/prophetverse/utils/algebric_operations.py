"""Functions that perform algebraic operations."""

from typing import Union

import jax.numpy as jnp
from numpy.typing import NDArray


def matrix_multiplication(
    data: Union[jnp.ndarray, NDArray], coefficients: Union[jnp.ndarray, NDArray]
) -> Union[jnp.ndarray, NDArray]:
    """Perform matrix multiplication between two matrixes.

    Parameters
    ----------
    data : Union[jnp.ndarray, NDArray]
        Array to be multiplied.
    coefficients : Union[jnp.ndarray, NDArray]
        Array of coefficients used at matrix multiplication.

    Returns
    -------
    Union[jnp.ndarray, NDArray]
        Matrix multiplication between data and coefficients.
    """
    return data @ coefficients.reshape((-1, 1))


def _exponent_safe(data: Union[jnp.ndarray, NDArray], exponent: float) -> jnp.ndarray:
    """Exponentiate an array without numerical errors replacing zeros with ones.

    Parameters
    ----------
    data : Union[jnp.ndarray, NDArray]
        Array to be exponentiated.
    exponent : float
        Expoent numerical value.

    Returns
    -------
    jnp.ndarray
        Exponentiated array.
    """
    exponent_safe = jnp.where(data == 0, 1, data) ** exponent
    return jnp.where(data == 0, 0, exponent_safe)
