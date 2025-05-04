"""Dataset for lifttest example."""

import numpy as np
import numpyro.distributions as dist
import pandas as pd

from prophetverse.effects import HillEffect, LinearEffect, LinearFourierSeasonality
from prophetverse.effects.trend import PiecewiseLinearTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.experimental.simulate import simulate
from prophetverse.sktime import Prophetverse
from prophetverse.utils.regex import exact, no_input_columns
from prophetverse.effects.target.univariate import NegativeBinomialTargetLikelihood


def get_index():
    """
    Generate a time index ranging from 2000-01-01 to 2005-01-01 with daily frequency.

    Returns
    -------
    pd.PeriodIndex
        The generated time index.
    """
    index = pd.period_range("2000-01-01", "2005-01-01", freq="D")
    return index


def get_X(index):
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
    rng = np.random.default_rng(0)

    X = pd.DataFrame(
        {
            "ad_spend_search": np.cumsum(rng.normal(0, 1, size=len(index))),
            # "ad_spend_social_media": np.cumsum(rng.normal(0, 1, size=len(index))),
        },
        index=index,
    )
    X -= X.min()
    X /= X.max()
    X += 0.05

    X["ad_spend_social_media"] = (
        X["ad_spend_search"] + 0.1 + rng.normal(0, 0.01, size=len(index))
    )
    X["ad_spend_social_media"] = X["ad_spend_social_media"] ** 3.5
    X *= 100_000

    return X


def get_groundtruth_model():
    """
    Define and configure a Prophetverse model with custom components.

    Returns
    -------
    Prophetverse
        Configured Prophetverse model.
    """
    model = Prophetverse(
        trend=PiecewiseLinearTrend(
            changepoint_interval=100,
            changepoint_prior_scale=1000,
            changepoint_range=-100,
            remove_seasonality_before_suggesting_initial_vals=False,
            offset_prior_loc=10000,
            global_rate_prior_loc=10000,
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
                "ad_spend_search",
                HillEffect(
                    half_max_prior=dist.Normal(20_000, 1000),
                    slope_prior=dist.Normal(3, 0.01),
                    max_effect_prior=dist.Normal(1e6, 1e-8),
                    effect_mode="additive",
                ),
                exact("ad_spend_search"),
            ),
            (
                "ad_spend_social_media",
                HillEffect(
                    half_max_prior=dist.Normal(10_000, 1000),
                    slope_prior=dist.Normal(1.5, 0.01),
                    max_effect_prior=dist.Normal(1e5, 1e-8),
                    effect_mode="additive",
                ),
                exact("ad_spend_social_media"),
            ),
        ],
        inference_engine=MAPInferenceEngine(
            num_steps=1, num_samples=1, progress_bar=True
        ),
        # likelihood=NegativeBinomialTargetLikelihood(noise_scale=0.05),
        scale=1,
    )

    return model


def get_samples(model, X):
    """
    Simulate samples from the provided model and input data.

    Parameters
    ----------
    model : Prophetverse
        The Prophetverse model to use for simulation.
    X : pd.DataFrame
        Exogenous data for the simulation.

    Returns
    -------
    dict
        Simulated samples from the model.
    """
    samples, model = simulate(model=model, fh=X.index, X=X, return_model=True)
    return samples, model


def get_y(samples, index):
    """
    Extract observed sales data from the simulated samples.

    Parameters
    ----------
    samples : dict
        Simulated samples from the model.
    index : pd.PeriodIndex
        Time index for the observed data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing observed sales data.
    """
    return samples.loc[0, "obs"].to_frame("sales")


def get_true_effect(samples, index):
    """
    Extract the true effects for the exogenous variables from the simulated samples.

    Parameters
    ----------
    samples : dict
        Simulated samples from the model.
    index : pd.PeriodIndex
        Time index for the true effects.

    Returns
    -------
    pd.DataFrame
        DataFrame containing true effects for the exogenous variables.
    """

    true_effect = samples.loc[0]
    return true_effect


def get_simulated_lift_test(X, model, samples, true_effect, n=10):
    """
    Perform a simulated lift test by perturbing exogenous variables.

    Parameters
    ----------
    X : pd.DataFrame
        Original exogenous data.
    model : Prophetverse
        The Prophetverse model to use for simulation.
    samples : dict
        Simulated samples from the model.
    true_effect : pd.DataFrame
        True effects of the exogenous variables.
    n : int, optional
        Number of samples to return for the lift test, by default 10.

    Returns
    -------
    tuple of pd.DataFrame
        Lift test results for each exogenous variable.
    """
    rng = np.random.default_rng(1)
    outs = []
    for col in ["ad_spend_search", "ad_spend_social_media"]:

        X_b = X.copy()

        X_b[col] = X_b[col] * rng.uniform(0.8, 1.5, size=X.shape[0])

        samples_b = model.predict_component_samples(X=X_b, fh=X.index)

        true_effect_b = samples_b.loc[0, [col]]

        lift = true_effect_b / true_effect

        lift_test_dataframe = pd.DataFrame(
            index=X.index,
            data={
                "lift": (lift[col] * rng.normal(1, 0.05)).clip(0, None),
                "x_start": X.loc[:, col],
                "x_end": X_b.loc[:, col],
            },
        )
        outs.append(lift_test_dataframe.sample(n=n, replace=False))

    return tuple(outs)


def get_dataset():
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
    index = get_index()
    X = get_X(index)
    model = get_groundtruth_model()
    samples, model = get_samples(model, X)
    y = get_y(samples, index)
    true_effect = get_true_effect(samples, index)
    lift_test = get_simulated_lift_test(X, model, samples, true_effect, n=30)

    return y, X, lift_test, true_effect
