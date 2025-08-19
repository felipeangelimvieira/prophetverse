"""Tests for Hurdle estimator."""

import numpy as np
import pytest
from sktime.datasets import load_PBS_dataset
from sktime.transformations.compose import YtoX
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.sktime.intermittent_demand import HurdleDemandForecaster


@pytest.mark.parametrize("family", ["poisson", "negative-binomial"])
@pytest.mark.parametrize("time_varying", [False, "rw"])
@pytest.mark.parametrize(
    "engine",
    [
        ("mcmc", {"num_samples": 5, "num_warmup": 50, "num_chains": 4}),
        ("map", {"num_steps": 10, "num_samples": 10}),
    ],
)
@pytest.mark.parametrize("external_regressor", [False, True])
def test_hurdle_model(
    family: str, time_varying: bool, engine, external_regressor: bool
):
    from skpro.distributions import Hurdle, NegativeBinomial, Poisson

    from prophetverse.engine import MAPInferenceEngine, MCMCInferenceEngine

    """Test that Hurdle model can be instantiated and run with default parameters."""
    engine_type, kwargs = engine

    if engine_type == "mcmc":
        engine = MCMCInferenceEngine(**kwargs)
    elif engine_type == "map":
        engine = MAPInferenceEngine(**kwargs)

    y = load_PBS_dataset()
    forecaster = HurdleDemandForecaster(
        family=family,
        time_varying_demand=time_varying,
        time_varying_probability=time_varying,
        inference_engine=engine,
    )

    if external_regressor:
        forecaster = (
            YtoX()
            ** FourierFeatures(
                sp_list=[12, 24], fourier_terms_list=[2, 3], keep_original_columns=False
            )
            ** forecaster
        )

    forecaster.fit(y)

    fh = np.arange(1, 5)
    y_pred = forecaster.predict(fh=fh)

    assert y_pred.shape == fh.shape

    y_dist = forecaster.predict_proba(fh=fh)

    assert isinstance(y_dist, Hurdle)

    if family == "poisson":
        assert isinstance(y_dist.distribution, Poisson)
    elif family == "negative-binomial":
        assert isinstance(y_dist.distribution, NegativeBinomial)
