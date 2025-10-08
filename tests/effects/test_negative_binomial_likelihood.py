import numpy as np
import pandas as pd
import pytest

from prophetverse.effects.trend.flat import FlatTrend
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import AdamOptimizer
from prophetverse import Prophetverse
from prophetverse.effects.target.univariate import (
    NegativeBinomialTargetLikelihood,
)


def make_negative_binomial_data(n_samples=100, n_series=1, r=5, p=0.4):
    """Generate synthetic count data for Negative Binomial likelihood testing."""
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    counts = np.random.negative_binomial(r, p, size=n_samples * n_series)

    if n_series == 1:
        index = dates
    else:
        index = pd.MultiIndex.from_product(
            [range(n_series), dates], names=["series", "time"]
        )

    y = pd.Series(counts, index=index).astype(int).to_frame("target")
    return y


def make_negative_binomial_features(y):
    """Generate features for the Negative Binomial likelihood test."""
    return pd.DataFrame(
        np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index
    )


@pytest.mark.smoke
def test_prophet_negative_binomial_basic():
    """Test basic functionality of Negative Binomial likelihood with synthetic data."""
    y = make_negative_binomial_data()
    X = make_negative_binomial_features(y)

    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(sp_list=[7], fourier_terms_list=[1], freq="D"),
                None,
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
        likelihood=NegativeBinomialTargetLikelihood(),
    )

    forecaster.fit(y.iloc[:-5], X.iloc[:-5])
    fh = list(range(1, 5))
    y_pred = forecaster.predict(X=X, fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh)
    assert y_pred.shape[1] == 1
    assert (y_pred >= 0).all().all()


@pytest.mark.smoke
def test_prophet_negative_binomial_hierarchical():
    """Test Negative Binomial likelihood with hierarchical data."""
    y = make_negative_binomial_data(n_samples=100, n_series=3)
    X = make_negative_binomial_features(y)

    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(sp_list=[7], fourier_terms_list=[1], freq="D"),
                None,
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
        likelihood=NegativeBinomialTargetLikelihood(),
    )

    dates = y.index.get_level_values(-1).unique()
    train_dates, test_dates = dates[:-5], dates[-5:]
    train_idx = y.index.get_level_values(-1).isin(train_dates)
    test_idx = y.index.get_level_values(-1).isin(test_dates)

    forecaster.fit(y.loc[train_idx], X.loc[train_idx])
    fh = list(range(1, 5))
    y_pred = forecaster.predict(X=X.loc[test_idx], fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * 3
    assert y_pred.shape[1] == 1
    assert (y_pred >= 0).all().all()


@pytest.mark.smoke
def test_prophet_negative_binomial_predict_methods():
    """Test additional prediction methods of Negative Binomial likelihood."""
    y = make_negative_binomial_data()
    X = make_negative_binomial_features(y)

    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            (
                "seasonality",
                LinearFourierSeasonality(sp_list=[7], fourier_terms_list=[1], freq="D"),
                None,
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(), num_steps=1, num_samples=1
        ),
        likelihood=NegativeBinomialTargetLikelihood(),
    )

    forecaster.fit(y.iloc[:-5], X.iloc[:-5])
    fh = list(range(1, 5))

    methods = [
        "predict_interval",
        "predict_components",
        "predict_component_samples",
        "predict_samples",
    ]

    for method in methods:
        preds = getattr(forecaster, method)(X=X, fh=fh)
        assert preds is not None
        assert isinstance(preds, pd.DataFrame)

        if method == "predict_interval":
            assert (preds >= 0).all().all()
