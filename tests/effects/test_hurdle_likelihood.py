import pandas as pd
import pytest
import numpyro
from numpyro import distributions as dist

from prophetverse.datasets.loaders import load_pedestrian_count
from prophetverse.effects.fourier import LinearFourierSeasonality
from prophetverse.effects.trend import FlatTrend
from prophetverse.effects.target.hurdle import HurdleTargetLikelihood
from prophetverse.sktime import Prophetverse
from prophetverse.engine import MAPInferenceEngine
from prophetverse.utils.regex import no_input_columns
from prophetverse.effects.constant import Constant


@pytest.mark.ci
def test_hurdle_target_likelihood_pedestrian_counts():
    """Integration test for HurdleTargetLikelihood on pedestrian count data.

    Uses a single sensor series with zero inflation to validate hurdle Poisson
    modeling end-to-end (fit + short-horizon predict) including zero probability
    branch effects.
    """
    numpyro.enable_x64()

    y = load_pedestrian_count()
    # Basic sanity
    assert "pedestrian_count" in y.columns

    # Zero-inflate small counts as in tutorial
    y = y.copy()
    y[y < 500] = 0

    # Select one series
    one = y.loc["T2"].copy()
    # Split: first year train (24*365 hours) then following portion as test
    split_index = 24 * 90  # shorter for test speed (~90 days)
    y_train = one.iloc[:split_index]
    y_test = one.iloc[split_index + 1 : split_index + 1 + 24 * 5]  # next 5 days

    # Ensure enough zeros present
    assert (y_train["pedestrian_count"] == 0).sum() > 0

    exogenous_effects = [
        (
            "seasonality",
            LinearFourierSeasonality(
                sp_list=[24, 24 * 7],  # daily + weekly
                fourier_terms_list=[2, 2],
                freq="H",
                prior_scale=0.1,
                effect_mode="multiplicative",
            ),
            no_input_columns,
        ),
        ("zero_proba__constant_term", Constant(prior=dist.Normal(0.5, 1)), None),
        (
            "zero_proba__seasonality",
            LinearFourierSeasonality(
                sp_list=[24, 24 * 7],
                fourier_terms_list=[2, 2],
                freq="H",
                prior_scale=0.1,
                effect_mode="additive",
            ),
            no_input_columns,
        ),
    ]

    model = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=exogenous_effects,
        inference_engine=MAPInferenceEngine(num_steps=100, progress_bar=False),
        likelihood=HurdleTargetLikelihood(likelihood_family="poisson"),
    )
    model.fit(y=y_train)

    fh = y_train.index[-48:].union(y_test.index[:48])  # last 2 days train + 2 days test
    preds = model.predict(fh=fh)

    # Shape & index alignment
    assert preds.shape[0] == len(fh)
    assert preds.shape[1] == 1
    # Non-negativity (counts support)
    assert (preds >= 0).all().item()

    # Basic sanity: forecast horizon covers both train tail and early test
    assert fh.min() >= y_train.index.min()
    assert fh.max() <= y_test.index.max()

    # Mean should not explode: enforce an upper bound relative to observed max in training subset
    assert preds.max().item() <= y_train["pedestrian_count"].max() * 5
