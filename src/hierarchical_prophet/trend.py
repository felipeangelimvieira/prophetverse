#  pylint: disable=g-import-not-at-top
from typing import Protocol, TypedDict, Dict, Tuple, Callable
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
import re
import logging
from abc import ABC, abstractmethod


class TrendStrategy(ABC):


    @abstractmethod
    def sample(self):
        raise NotImplementedError()

    @abstractmethod
    def get_trend(self, coefficients):
        ...
        
        
class PiecewiseLinearTrend(TrendStrategy):
    
    def __init__(self,
                 changepoint_coefficient_distribution,
                 offset_distribution,
                 changepoints_mask=None):
        self.changepoint_coefficient_distribution = changepoint_coefficient_distribution
        self.offset_distribution = offset_distribution
        self.changepoints_mask = changepoints_mask
    

         
