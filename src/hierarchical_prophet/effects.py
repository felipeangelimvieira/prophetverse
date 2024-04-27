#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict, Tuple, Callable
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
import re
import logging
from abc import ABC, abstractmethod


class AbstractEffect(ABC):

    def __init__(self, id=""):
        self.id = id

    
    def sample(self, name, *args, **kwargs):
        return numpyro.sample(f"{self.id}__{name}", *args, **kwargs)
    
    
    @abstractmethod
    def compute_effect(self, trend, data): ...

    def __call__(self, trend, data):

        return self.compute_effect(trend, data)


class LogEffect(AbstractEffect):

    def __init__(self, id=""):
        super().__init__(id)

    def compute_effect(self, trend, data):

        scale = self.sample("log_scale", dist.Gamma(concentration=1, rate=1))
        rate = self.sample("log_rate", dist.Gamma(concentration=1, rate=1))
        return scale * jnp.log(rate * data + 1)


class LinearEffect(AbstractEffect):

    def __init__(self, id="", dist = dist.Normal, dist_args=(0, 1), effect_mode="multiplicative"):
        self.dist = dist
        self.dist_args = dist_args
        self.effect_mode = effect_mode
        super().__init__(id)

    def compute_effect(self, trend, data):

        n_features = data.shape[-1]
        coefficients = self.sample(
            "coefs",
            self.dist(*[jnp.array([arg] * n_features) for arg in self.dist_args]),
        )
        
        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)
        if self.effect_mode == "multiplicative":
            return multiplicative_effect(trend, data, coefficients)
        return additive_effect(trend, data, coefficients)

class CustomPriorEffect(AbstractEffect):

    def __init__(self,
                     exogenous_priors,
                     feature_names,
                     default_exogenous_prior = (dist.Normal, 0, 1),
                     effect_mode = "multiplicative",
                     id=""):
        self.exogenous_priors = exogenous_priors
        self.feature_names = feature_names
        self.default_exogenous_prior = default_exogenous_prior
        self.effect_mode = effect_mode
        super().__init__(id)

        self.set_distributions_and_permutation_matrix()

    def compute_effect(self, trend, data):

        coefficients = self.get_coefficients()
        if self.effect_mode == "multiplicative":
            return multiplicative_effect(trend, data, coefficients)
        return additive_effect(trend, data, coefficients)

    @property
    def features_with_default_priors(self):
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
        for regex, _ in self.exogenous_priors.items():
            columns = [column for column in self.feature_names if re.match(regex, column)]
            if already_set_columns.intersection(columns):
                raise ValueError(
                        "Columns {} are already set".format(
                            already_set_columns.intersection(columns)
                        )
                    )
            already_set_columns = already_set_columns.union(columns)
        remaining_columns = self.feature_names.difference(already_set_columns)

        # Create a regex that matches all remaining columns
        remaining_regex = "|".join(remaining_columns)
        return remaining_regex

    def set_distributions_and_permutation_matrix(self):
        if self.feature_names is None or len(self.feature_names) == 0:
            return [], jnp.array([])

        exogenous_dists = []
        exogenous_permutation_matrix = []
        exogenous_priors = self.exogenous_priors.copy()

        if self.features_with_default_priors:
            exogenous_priors[self.features_with_default_priors] = self.default_exogenous_prior

        for i, (regex, (Distribution, *args)) in enumerate(exogenous_priors.items()):
            # Find columns that match the regex
            columns = [column for column in self.feature_names if re.match(regex, column)]
            # Get idx of columns that match the regex
            idx = jnp.array([self.feature_names.get_loc(column) for column in columns])
            # Set the distribution for each column that matches the regex
            distribution: dist.Distribution = Distribution(
                    *[jnp.ones(len(idx)) * arg for arg in args]
                )

            name = "exogenous_coefficients_{}".format(i)

            if not len(idx):
                logging.warning(
                        "No columns in the DataFrame match the regex pattern: {}".format(regex)
                    )
                continue
            # Matrix of shape (len(columns), len(idx) that map len(idx) to the corresponding indexes
            exogenous_permutation_matrix.append(jnp.eye(len(self.feature_names))[idx].T)
            exogenous_dists.append((name, distribution))

        self.exogenous_permutation_matrix = jnp.concatenate(exogenous_permutation_matrix, axis=1)
        self.exogenous_dists = exogenous_dists

    def get_coefficients(self):
        parameters = []
        for regex, distribution in self.exogenous_dists:
            parameters.append(
                    self.sample(
                        "exogenous_coefficients_{}".format(regex), distribution
                    ).reshape((-1, 1))
                )

        return self.exogenous_permutation_matrix @ jnp.concatenate(parameters, axis=0)


# --------------
#     Effects
# --------------


# Simple additive and multiplicative effects
# ------------------------------------------


def matrix_multiplication(data, coefficients):
    return data @ coefficients.reshape((-1, 1))


def additive_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:
    return matrix_multiplication(data, coefficients)


def multiplicative_effect(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: jnp.ndarray
) -> jnp.ndarray:
    return trend * matrix_multiplication(data, coefficients)


# Hill function
# -------------


def _apply_exponent_safe(
    data: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:
    """Applies an exponent to given data in a gradient safe way.

    More info on the double jnp.where can be found:
    https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    Args:
      data: Input data to use.
      exponent: Exponent required for the operations.

    Returns:
      The result of the exponent operation with the inputs provided.
    """
    exponent_safe = jnp.where(data == 0, 1, data) ** exponent
    return jnp.where(data == 0, 0, exponent_safe)


def hill(
    trend: jnp.ndarray, data: jnp.ndarray, coefficients: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Calculates the hill function for a given array of values.

    Refer to the following link for detailed information on this equation:
      https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

    Args:
      data: Input data.
      half_max_effective_concentration: ec50 value for the hill function.
      slope: Slope of the hill function.

    Returns:
      The hill values for the respective input data.
    """
    half_max_effective_concentration, slope = coefficients
    save_transform = _apply_exponent_safe(
        data=data / half_max_effective_concentration, exponent=-slope
    )
    return jnp.where(save_transform == 0, 0, 1.0 / (1 + save_transform))


def sample_hill_params(
    *args,
    half_max_concentration: float = 1,
    half_max_rate: float = 1,
    slope_concentration: float = 1,
    slope_rate: float = 1,
    **kwargs
):

    half_max_effective_concentration = numpyro.sample(
        "half_max_effective_concentration",
        dist.Gamma(concentration=half_max_concentration, rate=half_max_rate),
    )

    slope = numpyro.sample(
        "slope", dist.Gamma(concentration=slope_concentration, rate=slope_rate)
    )

    return (half_max_effective_concentration, slope)
