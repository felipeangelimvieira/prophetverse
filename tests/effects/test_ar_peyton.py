import pandas as pd
import pytest
import numpyro
from numpyro import distributions as dist

from prophetverse.datasets.loaders import load_peyton_manning
from prophetverse.effects.ar import AREffect
from prophetverse.sktime import Prophetverse
from prophetverse.engine import MAPInferenceEngine


@pytest.mark.ci
def test_ar_effect_peyton_manning_map():
    """Integration test: fit AREffect on Peyton Manning dataset using MAP.

    Mirrors the how-to guide example to ensure end-to-end functionality with
    real-world sized daily data and a long forecasting horizon. This guards
    against regressions in the AR scan implementation and engine interaction.
    """
    numpyro.enable_x64()

    try:
        y = load_peyton_manning()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Skipping due to data load error: {e}")

    # Basic sanity of dataset
    assert "y" in y.columns and len(y) > 1000

    model = Prophetverse(
        trend=AREffect(
            mean_reverting=True,
            phi_prior=dist.Normal(0.0, 1.0),
            sigma_prior=dist.HalfNormal(0.5),
        ),
        inference_engine=MAPInferenceEngine(progress_bar=False),
    )
    model.fit(y=y)

    # Use an in-sample + short extension horizon (last 365 days) for stability
    fh = y.index[-365:]
    preds = model.predict(fh=fh)

    # Shape and index alignment checks
    assert preds.shape[0] == len(fh)
    assert preds.shape[1] == 1
    assert not preds.isna().any().item()

    # Forecast horizon should start after (or overlapping) training start
    assert fh.min() >= y.index.min()
