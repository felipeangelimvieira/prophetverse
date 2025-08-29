import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed
import pytest

from prophetverse.effects.ar import AREffect


@pytest.fixture
def sample_series():
    idx = pd.date_range(start="2024-01-01", periods=5, freq="D")
    # y only needed for index; values arbitrary
    y = pd.DataFrame({"y": jnp.arange(5)}, index=idx)
    return y


def test_ar_effect_shape_mean_reverting(sample_series):
    effect = AREffect(mean_reverting=True)
    effect.fit(y=sample_series, X=None, scale=1.0)

    fh = pd.date_range(start="2024-01-06", periods=3, freq="D")
    data = effect.transform(X=None, fh=fh)
    with seed(numpyro.handlers.seed, 0):
        out = effect.predict(data=data, predicted_effects={})
    # Should return len(fh) rows, 1 column
    assert out.shape == (len(fh), 1)


def test_ar_effect_shape_random_walk(sample_series):
    effect = AREffect(mean_reverting=False)
    effect.fit(y=sample_series, X=None, scale=1.0)
    fh = pd.date_range(start="2024-01-03", periods=2, freq="D")  # overlaps train
    data = effect.transform(X=None, fh=fh)
    with seed(numpyro.handlers.seed, 1):
        out = effect.predict(data=data, predicted_effects={})
    assert out.shape == (len(fh), 1)


def test_ar_effect_internal_full_sampling(sample_series):
    # Forecast beyond training end ensures full path union logic exercised
    effect = AREffect(mean_reverting=True)
    effect.fit(y=sample_series, X=None, scale=1.0)
    fh = pd.date_range(start="2024-01-08", periods=2, freq="D")
    data = effect.transform(X=None, fh=fh)
    placeholder, ix = data
    # Full index should span from earliest train to latest fh
    assert placeholder.shape[0] == 5 + 2 + 2  # 5 training days + gap (6,7) + 2 fh
    # Index mapping length equals fh length
    assert ix.shape[0] == len(fh)
