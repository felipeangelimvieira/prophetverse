#  pylint: disable=g-import-not-at-top
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Protocol, Tuple, TypedDict

import jax.numpy as jnp
import numpyro
import pandas as pd
from numpyro import distributions as dist

# --------------
#     Effects
# --------------

class AbstractEffect(ABC):
    """Abstract class for effects.
    
    Effects should inherit from this class and implement the `compute_effect` method.
    The id is used to create unique names for the samples in the model.
    
    """

    def __init__(self, id="", regex=None, **kwargs):
        self.id = id
        self.regex = regex
        
        
    def match_columns(self, columns : pd.Index) -> pd.Index:
        """Match the columns of the DataFrame with the regex pattern.
        
        Args:
            X (pd.DataFrame): The DataFrame to match.
        
        Returns:
            pd.Index: The columns that match the regex pattern.
        """
        
        if isinstance(columns, List):
            columns = pd.Index(columns)
            
        if self.regex is None:
            raise ValueError("To use this method, you must set the regex pattern")
        return columns[columns.str.match(self.regex)]
    
    @staticmethod
    def split_data_into_effects(X : pd.DataFrame, effects : List) -> Dict[str, pd.DataFrame]:
        """Split the data into effects.
        
        Args:
            X (pd.DataFrame): The DataFrame to split.
            effects (List[AbstractEffect]): The effects to split the data into.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping effect names to DataFrames.
        """
        data = {}
        for effect in effects:
            data[effect.id] = X[effect.match_columns(X.columns)]
        return data

    def sample(self, name : str, *args, **kwargs):
        """
        Sample a random variable with a unique name.
        """
        return numpyro.sample(f"{self.id}__{name}", *args, **kwargs)

    @abstractmethod
    def compute_effect(self, trend : jnp.ndarray, data : jnp.ndarray) -> jnp.ndarray: 
        """Compute the effect based on the trend and data.
        
        Args:
            trend (jnp.ndarray): The trend.
            data (jnp.ndarray): The data concerning this effect.
        
        Returns:
            jnp.ndarray: The effect.
        """
        ...

    def __call__(self, trend, data):

        return self.compute_effect(trend, data)


class LogEffect(AbstractEffect):
    """
    Log effect for a variable.

    Computes the effect using the formula:

    effect = scale * log(rate * data + 1)

    A gamma prior is used for the scale and rate parameters.

    Args:
        id (str): The identifier for the effect.
        scale_prior (dist.Distribution): The prior distribution for the scale parameter.
        rate_prior (dist.Distribution): The prior distribution for the rate parameter.
    """

    def __init__(
        self,
        scale_prior=None,
        rate_prior=None,
        effect_mode="multiplicative",
        **kwargs,
    ):
        if scale_prior is None:
            scale_prior = dist.Gamma(1, 1)
        if rate_prior is None:
            rate_prior = dist.Gamma(1, 1)
            
        self.scale_prior = scale_prior
        self.rate_prior = rate_prior
        self.effect_mode = effect_mode
        super().__init__(**kwargs)

    def compute_effect(self, trend, data):
        """
        Computes the effect using the log transformation.

        Args:
            trend: The trend component.
            data: The input data.

        Returns:
            The computed effect.
        """
        scale = self.sample("log_scale", self.scale_prior)
        rate = self.sample("log_rate", self.rate_prior)
        effect = scale * jnp.log(rate * data + 1)
        if self.effect_mode == "additive":
            return effect
        return trend * effect


class LinearEffect(AbstractEffect):
    """
    Represents a linear effect in a hierarchical prophet model.

    Args:
        id (str): The identifier for the effect.
        prior (tuple): A tuple with the distribution class to use for sampling coefficients and  the arguments to pass to the distribution class. Defaults to (dist.Normal, 0, 1).
        effect_mode (str): The mode of the effect, either "multiplicative" or "additive".

    Attributes:
        dist (type): The distribution class used for sampling coefficients.
        dist_args (tuple): The arguments passed to the distribution class.
        effect_mode (str): The mode of the effect, either "multiplicative" or "additive".

    Methods:
        compute_effect(trend, data): Computes the effect based on the given trend and data.

    """

    def __init__(
        self,
        prior=(dist.Normal, 0, 1),
        effect_mode="multiplicative",
        **kwargs):
        self.prior = prior
        self.effect_mode = effect_mode
        super().__init__(**kwargs)

    def compute_effect(self, trend, data):
        """
        Computes the effect based on the given trend and data.

        Args:
            trend: The trend component of the hierarchical prophet model.
            data: The data used to compute the effect.

        Returns:
            The computed effect based on the given trend and data.

        """
        n_features = data.shape[-1]

        dist = self.prior[0]
        dist_args = self.prior[1:]
        coefficients = self.sample(
            "coefs",
            dist(*[jnp.array([arg] * n_features) for arg in dist_args]),
        )

        if coefficients.ndim == 1:
            coefficients = jnp.expand_dims(coefficients, axis=-1)

        if data.ndim == 3 and coefficients.ndim == 2:
            coefficients = jnp.expand_dims(coefficients, axis=0)
        if self.effect_mode == "multiplicative":
            return multiplicative_effect(trend, data, coefficients)
        return additive_effect(trend, data, coefficients)

class HillEffect(AbstractEffect):
    """
    Represents a Hill effect in a time series model.

    Attributes:
        half_max_prior: Prior distribution for the half-maximum parameter.
        slope_prior: Prior distribution for the slope parameter.
        max_effect_prior: Prior distribution for the maximum effect parameter.
        effect_mode: Mode of the effect (either "additive" or "multiplicative").
    """

    def __init__(
        self,
        half_max_prior=None,
        slope_prior=None,
        max_effect_prior=None,
        effect_mode="multiplicative",
        **kwargs,
    ):
        
        if half_max_prior is None:
            half_max_prior = dist.Gamma(1, 1)
        if slope_prior is None:
            slope_prior = dist.HalfNormal(10)
        if max_effect_prior is None:
            max_effect_prior = dist.Gamma(1, 1)
            
        self.half_max_prior = half_max_prior
        self.slope_prior = slope_prior
        self.max_effect_prior = max_effect_prior
        self.effect_mode = effect_mode
        super().__init__(**kwargs)

    def compute_effect(self, trend, data):
        """
        Computes the effect using the log transformation.

        Args:
            trend: The trend component.
            data: The input data.

        Returns:
            The computed effect.
        """

        half_max = self.sample("half_max", self.half_max_prior)
        slope = self.sample("slope", self.slope_prior)
        max_effect = self.sample("max_effect", self.max_effect_prior)

        x = _exponent_safe(data / half_max, -slope)
        effect = max_effect / (1 + x)

        if self.effect_mode == "additive":
            return effect
        return trend * effect


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


def _exponent_safe(data, exponent):
    # From lightweight mmm library
    exponent_safe = jnp.where(data == 0, 1, data) ** exponent
    return jnp.where(data == 0, 0, exponent_safe)
