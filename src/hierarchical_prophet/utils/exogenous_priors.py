import re
from typing import Dict, List, Tuple

import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as jnp
from jax import random
from hierarchical_prophet.logger import logger


def get_exogenous_priors(
    X: pd.DataFrame,
    exogenous_priors: Dict[str, Tuple[dist.Distribution, ...]],
    default_exogenous_prior=None,
) -> Tuple[Tuple[str, dist.Distribution], jnp.ndarray]:
    """
    Get the exogenous priors for each column in the input DataFrame.

    This function takes an input DataFrame `X` and a dictionary `exogenous_priors` that maps regular expressions to tuples of distributions and their arguments.
    It returns a tuple containing the exogenous distributions and the permutation matrix.

    Parameters:
        X (pd.DataFrame): The input DataFrame.
        exogenous_priors (Dict[str, Tuple[dist.Distribution, ...]]): A dictionary mapping regular expressions to tuples of distributions and their arguments.
        default_exogenous_prior (Optional): The default exogenous prior to use for columns that do not match any regular expression.

    Returns:
        Tuple[Tuple[str, dist.Distribution], jnp.ndarray]: A tuple containing the exogenous distributions and the permutation matrix.
    """
    
    if X is None or X.columns.empty:
        return [], jnp.array([])
    
    exogenous_dists = []

    exogenous_permutation_matrix = []
    remaining_columns_regex = check_regexes_and_return_remaining_regex(
        exogenous_priors, X
    )

    exogenous_priors = exogenous_priors.copy()

    if remaining_columns_regex:
        exogenous_priors[remaining_columns_regex] = default_exogenous_prior

    for i, (regex, (Distribution, *args)) in enumerate(exogenous_priors.items()):
        # Find columns that match the regex
        columns = [column for column in X.columns if re.match(regex, column)]
        # Get idx of columns that match the regex
        idx = jnp.array([X.columns.get_loc(column) for column in columns])
        # Set the distribution for each column that matches the regex
        distribution: dist.Distribution = Distribution(
            *[jnp.ones(len(idx)) * arg for arg in args]
        )

        name = "exogenous_coefficients_{}".format(i)

        if not len(idx):
            logger.warning(
                "No columns in the DataFrame match the regex pattern: {}".format(regex)
            )
            continue
        # Matrix of shape (len(columns), len(idx) that map len(idx) to the corresponding indexes
        exogenous_permutation_matrix.append(jnp.eye(len(X.columns))[idx].T)
        exogenous_dists.append((name, distribution))

    exogenous_permutation_matrix = jnp.concatenate(exogenous_permutation_matrix, axis=1)

    return exogenous_dists, exogenous_permutation_matrix


def sample_exogenous_coefficients(exogenous_dists, exogenous_permutation_matrix):
    """
    Sample exogenous coefficients based on the given exogenous distributions and permutation matrix.

    This function takes the exogenous distributions and permutation matrix obtained from `get_exogenous_priors` function.
    It samples the exogenous coefficients using the distributions and returns the result.

    Parameters:
        exogenous_dists (List[Tuple[str, dist.Distribution]]): A list of tuples containing the name and distribution of exogenous coefficients.
        exogenous_permutation_matrix (jnp.ndarray): The permutation matrix obtained from `get_exogenous_priors` function.

    Returns:
        jnp.ndarray: The sampled exogenous coefficients.
    """
    parameters = []
    for regex, distribution in exogenous_dists:
        parameters.append(
            numpyro.sample(
                "exogenous_coefficients_{}".format(regex), distribution
            ).reshape((-1, 1))
        )

    return exogenous_permutation_matrix @ jnp.concatenate(parameters, axis=0)


def check_regexes_and_return_remaining_regex(
    exogenous_priors: List[Tuple], X: pd.DataFrame
) -> str:
    """
    Check the regular expressions in the exogenous priors and return the remaining regex.

    This function takes the exogenous priors and the input DataFrame `X`.
    It checks if any columns are already set based on the regular expressions in the exogenous priors.
    It returns the remaining regex that matches the columns that are not already set.

    Parameters:
        exogenous_priors (List[Tuple]): The exogenous priors.
        X (pd.DataFrame): The input DataFrame.

    Returns:
        str: The remaining regex that matches the columns that are not already set.
    """
    already_set_columns = set()
    for regex, _ in exogenous_priors.items():
        columns = [column for column in X.columns if re.match(regex, column)]
        if already_set_columns.intersection(columns):
            raise ValueError(
                "Columns {} are already set".format(
                    already_set_columns.intersection(columns)
                )
            )
        already_set_columns = already_set_columns.union(columns)
    remaining_columns = X.columns.difference(already_set_columns)

    # Create a regex that matches all remaining columns
    remaining_regex = "|".join(remaining_columns)
    return remaining_regex
