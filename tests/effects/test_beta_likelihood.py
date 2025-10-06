import numpy as np
import pandas as pd
import pytest
from numpyro import distributions as dist

from prophetverse.effects.trend.flat import FlatTrend
from prophetverse.effects import LinearFourierSeasonality
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import AdamOptimizer
from prophetverse import Prophetverse
from prophetverse.effects.target.univariate import BetaTargetLikelihood


def make_beta_data(n_samples=100, n_series=1):
    """Generate synthetic data in [0,1] range for Beta likelihood testing."""
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    if n_series == 1:
        index = dates
    else:
        index = pd.MultiIndex.from_product(
            [range(n_series), dates], names=["series", "time"]
        )

    # Generate data between 0 and 1
    y = pd.Series(
        np.random.beta(a=2, b=5, size=n_samples * n_series), index=index
    ).to_frame("target")
    return y


def make_beta_features(y):
    """Generate features for the Beta likelihood test."""
    return pd.DataFrame(
        np.random.rand(len(y), 3), columns=["x1", "x2", "x3"], index=y.index
    )


@pytest.mark.smoke
def test_prophet_beta_basic():
    """Test basic functionality of ProphetBeta with synthetic data."""
    y = make_beta_data()
    X = make_beta_features(y)

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
        likelihood=BetaTargetLikelihood(),
    )

    # Test fit and predict
    forecaster.fit(y.iloc[:-5], X.iloc[:-5])
    fh = list(range(1, 5))
    y_pred = forecaster.predict(X=X, fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh)
    assert y_pred.shape[1] == 1
    assert all((y_pred >= 0) & (y_pred <= 1))  # Predictions should be in [0,1]


@pytest.mark.smoke
def test_prophet_beta_hierarchical():
    """Test ProphetBeta with hierarchical data."""
    y = make_beta_data(n_samples=100, n_series=3)
    X = make_beta_features(y)

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
        likelihood=BetaTargetLikelihood(),
    )

    dates = y.index.get_level_values(-1).unique()
    train_dates, test_dates = dates[:-5], dates[-5:]
    train_idx = y.index.get_level_values(-1).isin(train_dates)
    test_idx = y.index.get_level_values(-1).isin(test_dates)
    # Test fit and predict
    forecaster.fit(y.loc[train_idx], X.loc[train_idx])
    fh = list(range(1, 5))
    y_pred = forecaster.predict(X=X.loc[test_idx], fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * 3  # 3 series
    assert y_pred.shape[1] == 1
    assert all((y_pred >= 0) & (y_pred <= 1))  # Predictions should be in [0,1]


@pytest.mark.smoke
def test_prophet_beta_predict_methods():
    """Test additional prediction methods of ProphetBeta."""
    y = make_beta_data()
    X = make_beta_features(y)

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
        likelihood=BetaTargetLikelihood(),
    )

    forecaster.fit(y.iloc[:-5], X.iloc[:-5])
    fh = list(range(1, 5))

    # Test different prediction methods
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
            assert all((preds >= 0) & (preds <= 1))  # Intervals should be in [0,1]
