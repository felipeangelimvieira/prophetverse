from typing import Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor

from .base import TrendModel


class FlatTrend(TrendModel):
    

    def __init__(self, changepoint_prior_scale=0.1) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        super().__init__()

    def initialize(self, y: pd.DataFrame):

        self.changepoint_prior_loc = y.mean().values

    def prepare_input_data(self, idx: pd.PeriodIndex) -> dict:

        return {
            "constant_vector": jnp.ones((len(idx), 1)),
        }
        
    def compute_trend(self, constant_vector):
        
        mean =self.changepoint_prior_loc
        var = self.changepoint_prior_scale**2
        
        rate = mean / var
        concentration = mean * rate 
        
        coefficient = numpyro.sample(
            "trend_flat_coefficient",
            dist.Gamma(
                rate=rate,
                concentration=concentration,
            )
        )
        
        return constant_vector * coefficient
