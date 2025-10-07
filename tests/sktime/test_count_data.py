# Disable warnings
import warnings

warnings.simplefilter(action="ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.fourier import FourierFeatures


from prophetverse.datasets.loaders import load_pedestrian_count

import numpyro

numpyro.enable_x64()

y = load_pedestrian_count()

# We take only one time series for simplicity
y = y.loc["T2"]

split_index = 24 * 365
y_train, y_test = y.iloc[:split_index], y.iloc[split_index + 1 : split_index * 2 + 1]


from prophetverse.effects.fourier import LinearFourierSeasonality
from prophetverse.effects.trend import FlatTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import CosineScheduleAdamOptimizer, LBFGSSolver

from prophetverse.sktime import Prophetverse
from prophetverse.utils.regex import no_input_columns

# Here we set the prior for the seasonality effect
# And the coefficients for it
exogenous_effects = [
    (
        "seasonality",
        LinearFourierSeasonality(
            sp_list=[24, 24 * 7, 24 * 365.5],
            fourier_terms_list=[2, 2, 10],
            freq="H",
            prior_scale=0.1,
            effect_mode="multiplicative",
        ),
        no_input_columns,
    ),
]

model = Prophetverse(
    trend=FlatTrend(),
    exogenous_effects=[],  # exogenous_effects,
    inference_engine=MAPInferenceEngine(),
)

model.set_params(likelihood="negbinomial")
model.fit(y=y_train)
