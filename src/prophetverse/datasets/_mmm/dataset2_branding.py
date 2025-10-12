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
    MultiplyEffects,
    Constant,
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
                "latent/awareness",
                HillEffect(effect_mode="additive"),
                exact("ad_spend_awareness"),
            ),
            (
                "latent/awareness_adstock",
                ChainedEffects(
                    steps=[
                        ("latent_awareness", Forward("latent/awareness")),
                        ("adstock", WeibullAdstockEffect(max_lag=90)),
                    ]
                ),
                None,
            ),
            (
                "latent/baseline",
                SumEffects(
                    effects=[
                        ("trend", Forward("trend")),
                        ("yearly_seasonality", Forward("yearly_seasonality")),
                        ("weekly_seasonality", Forward("weekly_seasonality")),
                        ("monthly_seasonality", Forward("monthly_seasonality")),
                        ("latent_awareness", Forward("latent/awareness_adstock")),
                    ]
                ),
                None,
            ),
            (
                "awareness_to_sales",
                MultiplyEffects(
                    [
                        ("linear", Constant()),
                        ("awareness", Forward("latent/awareness_adstock")),
                    ]
                ),
                None,
            ),
            (
                "last_click_spend",
                MultiplyEffects(
                    [
                        (
                            "hill",
                            ChainedEffects(
                                steps=[
                                    ("adstock", GeometricAdstockEffect()),
                                    ("saturation", HillEffect(effect_mode="additive")),
                                ]
                            ),
                        ),
                        ("baseline", Forward("latent/baseline")),
                    ]
                ),
                exact("last_click_spend"),
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

    X = get_X(index, rng, power=1)
    X = X.rename(
        columns={
            "ad_spend_search": "ad_spend_awareness",
            "ad_spend_social_media": "last_click_spend",
        }
    )
    # Select 20 days to add dummy campaigns
    return X


def get_dataset(return_y_and_X_only=False):
    """
    Generate a complete dataset.

    Includes time index, exogenous data, model,
    simulated samples, observed sales, true effects, and lift test results.

    We simulate a usecase where we have two channels:
    1. A last click channel (e.g. social media) that has direct impact
         on sales and is correlated with the latent awareness channel.
    2. A latent channel (e.g. search ads) that has no direct impact on sales
         but drives awareness and therefore has an indirect impact on sales.

    Returns
    -------
    tuple
        Contains observed sales, exogenous data, lift test results, true effects,
        and the Prophetverse model.
    """

    rng = np.random.default_rng(0)
    index = get_index()
    X = _get_X(index, rng=rng)

    # We will fit, predict, and then use the prediction to create the
    # investment in last click to simulate correlation
    model = get_groundtruth_model()

    model.fit(X=X, y=pd.Series(np.zeros(X.shape[0]), index=X.index))
    y = model.predict(X=X, fh=index)

    X["last_click_spend"] = X["last_click_spend"].mean() / y.mean() * y

    model.fit(X=X, y=pd.Series(np.zeros(X.shape[0]), index=X.index))
    y = model.predict(X=X, fh=index)

    true_effect = model.predict_components(X=X, fh=index)

    X, y = X.iloc[90:], y.iloc[90:]
    true_effect = true_effect.iloc[90:]
    if return_y_and_X_only:
        return y, X
    return y, X, true_effect, model
