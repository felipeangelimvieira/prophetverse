import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Protocol, Tuple, TypedDict
from prophetverse.effects.base import AbstractEffect
import jax.numpy as jnp
import numpyro
import pandas as pd
from numpyro import distributions as dist

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
 