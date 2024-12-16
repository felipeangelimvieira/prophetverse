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
            "investment1": np.cumsum(rng.normal(0, 1, size=len(index))),
            "investment2": np.cumsum(rng.normal(0, 1, size=len(index))),
        },
        index=index,
    )
    X -= X.min()
    X /= X.max()
    X += 0.05

    X["investment2"] = X["investment1"] + 0.1 + rng.normal(0, 0.01, size=len(index))
    X.plot.line(alpha=0.9)

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
            changepoint_prior_scale=0.001,
            changepoint_range=-100,
        ),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(
                    freq="D",
                    sp_list=[365.25],
                    fourier_terms_list=[3],
                    prior_scale=0.1,
                    effect_mode="multiplicative",
                ),
                no_input_columns,
            ),
            (
                "investment1",
                HillEffect(
                    half_max_prior=dist.HalfNormal(0.2),
                    slope_prior=dist.Gamma(2, 1),
                    max_effect_prior=dist.HalfNormal(1.5),
                    effect_mode="additive",
                ),
                exact("investment1"),
            ),
            (
                "investment2",
                LinearEffect(
                    prior=dist.HalfNormal(0.5),
                    effect_mode="additive",
                ),
                exact("investment2"),
            ),
        ],
        inference_engine=MAPInferenceEngine(num_steps=1),
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
    samples = simulate(
        model=model,
        fh=X.index,
        X=X,
    )
    return samples


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
    return pd.DataFrame(data={"sales": samples["obs"][0].flatten()}, index=index)


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
    true_effect = pd.DataFrame(
        data={
            "investment1": samples["investment1"][0].flatten(),
            "investment2": samples["investment2"][0].flatten(),
        },
        index=index,
    )
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

    X_b = X.copy()

    for col in ["investment1", "investment2"]:

        X_b[col] = X_b[col] * rng.uniform(0.1, 0.9, size=X.shape[0])

    samples_b = simulate(
        model=model.clone().set_params(inference_engine__num_steps=1),
        fh=X.index,
        X=X_b,
        do={k: v[0] for k, v in samples.items()},
    )

    true_effect_b = pd.DataFrame(
        index=X_b.index,
        data={
            "investment1": samples_b["investment1"][0].flatten(),
            "investment2": samples_b["investment2"][0].flatten(),
        },
    )

    lift = np.abs(true_effect_b - true_effect)

    outs = []

    for col in ["investment1", "investment2"]:
        lift_test_dataframe = pd.DataFrame(
            index=X.index,
            data={
                "lift": lift[col],
                "x_start": X.loc[:, col],
                "x_end": X_b.loc[:, col],
                "y_start": true_effect.loc[:, col],
                "y_end": true_effect_b.loc[:, col],
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
    samples = get_samples(model, X)
    y = get_y(samples, index)
    true_effect = get_true_effect(samples, index)
    lift_test = get_simulated_lift_test(X, model, samples, true_effect, n=10)

    return y, X, lift_test, true_effect, model
