"""Defines the application of effects: additive or multiplicative."""

from typing import Literal

import jax.numpy as jnp

from prophetverse.utils.algebric_operations import matrix_multiplication

__all__ = ["additive_effect", "multiplicative_effect"]

effects_application = Literal["additive", "multiplicative"]


def additive_effect(data: jnp.ndarray, coefficients: jnp.ndarray) -> jnp.ndarray:
    """Apply an additive effect.

    Parameters
    ----------
    data : jnp.ndarray
        Data without the effect.
    coefficients : jnp.ndarray
        Effect vector with coefficients of each column.

    Returns
    -------
    jnp.ndarray
        Data with the effect applied.
    """
    return matrix_multiplication(data, coefficients)


def multiplicative_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:
    """Apply a multiplicative effect.

    Parameters
    ----------
    trend : jnp.ndarray
        Trend coefficient.
    data : jnp.ndarray
        Data without the effect.
    coefficients : jnp.ndarray
        Effect vector with coefficients of each column.

    Returns
    -------
    jnp.ndarray
        Data with the effect applied.
    """
    return trend * matrix_multiplication(data, coefficients)
