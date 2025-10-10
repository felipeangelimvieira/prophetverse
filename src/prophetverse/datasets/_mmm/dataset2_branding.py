"""Dataset for branding example"""

import numpy as np
from prophetverse.datasets._mmm.dataset1 import get_index, get_X
import numpyro.distributions as dist
import pandas as pd
from pathlib import Path
from prophetverse.effects import (
    HillEffect,
    LinearEffect,
    LinearFourierSeasonality,
    ChainedEffects,
    GeometricAdstockEffect,
    WeibullAdstockEffect,
    SumEffects,
    Forward,
)
from prophetverse.effects.trend import PiecewiseLinearTrend
from prophetverse.engine.prior import PriorPredictiveInferenceEngine
from prophetverse.experimental.simulate import simulate
from prophetverse.sktime import Prophetverse
from prophetverse.utils.regex import exact, no_input_columns
from prophetverse.effects.target.univariate import NegativeBinomialTargetLikelihood
import jax.numpy as jnp
import json


def get_site_values():

    posterior_samples_path = Path(__file__).parent / Path(
        "dataset2_branding_posterior_samples.json"
    )
    with open(posterior_samples_path, "r") as f:
        posterior_samples = json.load(f)

    for key in posterior_samples.keys():
        posterior_samples[key] = jnp.array(posterior_samples[key])

    return posterior_samples


def get_groundtruth_model():
    """
    Define and configure a Prophetverse model with custom components.

    Returns
    -------
    Prophetverse
        Configured Prophetverse model.
    """

    site_samples = get_site_values()
    model = Prophetverse(
        trend=PiecewiseLinearTrend(
            changepoint_interval=100,
            changepoint_range=-100,
            remove_seasonality_before_suggesting_initial_vals=False,
        ),
        exogenous_effects=[
            (
                "yearly_seasonality",
                LinearFourierSeasonality(
                    freq="D",
                    sp_list=[365.25],
                    fourier_terms_list=[3],
                    prior_scale=0.05,
                    effect_mode="multiplicative",
                ),
                no_input_columns,
            ),
            (
                "weekly_seasonality",
                LinearFourierSeasonality(
                    freq="D",
                    sp_list=[7],
                    fourier_terms_list=[3],
                    prior_scale=0.01,
                    effect_mode="multiplicative",
                ),
                no_input_columns,
            ),
            (
                "monthly_seasonality",
                LinearFourierSeasonality(
                    freq="D",
                    sp_list=[28],
                    fourier_terms_list=[5],
                    prior_scale=0.05,
                    effect_mode="multiplicative",
                ),
                no_input_columns,
            ),
            (
                "latent/ad_spend_awareness",
                HillEffect(effect_mode="additive"),
                exact("ad_spend_awareness"),
            ),
            (
                "latent/awareness_campaign",
                ChainedEffects(
                    steps=[
                        ("linear", LinearEffect(effect_mode="additive")),
                        ("adstock", WeibullAdstockEffect(max_lag=90)),
                    ]
                ),
                exact("awareness_campaign"),
            ),
            (
                "latent/awareness",
                SumEffects(
                    effects=[
                        ("ad_spend_awareness", Forward("latent/ad_spend_awareness")),
                        ("awareness_campaign", Forward("latent/awareness_campaign")),
                    ]
                ),
                None,
            ),
            (
                "awareness_to_sales",
                ChainedEffects(
                    steps=[
                        ("latent_awareness", Forward("latent/awareness")),
                        ("adstock", WeibullAdstockEffect(max_lag=90)),
                    ]
                ),
                None,
            ),
            (
                "ad_spend_social_media",
                ChainedEffects(
                    steps=[
                        ("adstock", GeometricAdstockEffect()),
                        ("saturation", HillEffect(effect_mode="additive")),
                    ]
                ),
                exact("ad_spend_social_media"),
            ),
        ],
        inference_engine=PriorPredictiveInferenceEngine(
            num_samples=1, substitute=site_samples
        ),
        scale=1,
    )
    return model


def _get_X(index, rng):
    """
    Create a DataFrame of two simulated investments with daily data.

    Parameters
    ----------
    index : pd.PeriodIndex
        The time index for the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing normalized simulated investments.
    """

    X = get_X(index, rng)
    X = X.rename(columns={"ad_spend_search": "ad_spend_awareness"})
    # Select 20 days to add dummy campaigns
    idxs = rng.choice(np.arange(30, len(index) - 30), size=20, replace=False)
    X.loc[X.index[idxs], "awareness_campaign"] = 1.0
    X["awareness_campaign"] = X["awareness_campaign"].fillna(0)
    return X


def get_dataset(return_y_and_X_only=False):
    """
    Generate a complete dataset.

    Includes time index, exogenous data, model,
    simulated samples, observed sales, true effects, and lift test results.

    Returns
    -------
    tuple
        Contains observed sales, exogenous data, lift test results, true effects,
        and the Prophetverse model.
    """

    rng = np.random.default_rng(0)
    index = get_index()
    X = _get_X(index, rng=rng)
    model = get_groundtruth_model()

    model.fit(X=X, y=pd.Series(np.zeros(X.shape[0]), index=X.index))
    y = model.predict(X=X, fh=index)
    true_effect = model.predict_components(X=X, fh=index)

    X, y = X.iloc[90:], y.iloc[90:]
    true_effect = true_effect.iloc[90:]
    if return_y_and_X_only:
        return y, X
    return y, X, true_effect, model
