"""Utils for inference engines."""

from typing import Dict

import jax.numpy as jnp
import numpy as np

from prophetverse.exc import ConvergenceError


def assert_mcmc_converged(summary: Dict[str, Dict[str, jnp.ndarray]], max_r_hat: float):
    """Assert that an MCMC program has converged.

    Parameters
    ----------
    summary: Dict
        MCMC trace summary.

    max_r_hat: float
        Maximum allowed r_hat.

    Returns
    -------
        Nothing.

    Raises
    ------
    ConvergenceError
    """
    for name, parameter_summary in summary.items():
        # NB: some variables have deterministic elements (s.a. samples from LKJCov).
        mask = np.isnan(parameter_summary["n_eff"])
        r_hat = parameter_summary["r_hat"][~mask]

        if (r_hat <= max_r_hat).all():
            continue

        # TODO: might be better to print entire
        #  summary instead of just which parameter didn't converge
        raise ConvergenceError(f"Parameter '{name}' did not converge! R_hat: {r_hat}")

    return
