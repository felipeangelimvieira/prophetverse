import pandas as pd
import numpy as np
import numpyro
from numpyro.handlers import seed

from prophetverse.effects.ar import AREffect


def test_ar_effect_period_index_mean_reverting():
    periods = pd.period_range(start="2024-01", periods=6, freq="M")
    y = pd.DataFrame({"y": np.arange(len(periods))}, index=periods)

    effect = AREffect(mean_reverting=True)
    effect.fit(y=y, X=None, scale=1.0)

    fh = pd.period_range(start="2024-07", periods=3, freq="M")
    data = effect.transform(X=None, fh=fh)
    placeholder, ix = data
    assert placeholder.shape[0] >= len(y) + len(fh)
    assert ix.shape[0] == len(fh)
    with seed(numpyro.handlers.seed, 0):
        out = effect.predict(data=data, predicted_effects={})
    assert out.shape == (len(fh), 1)


def test_ar_effect_period_index_random_walk_overlap():
    periods = pd.period_range(start="2024-01", periods=4, freq="Q")
    y = pd.DataFrame({"y": np.arange(len(periods))}, index=periods)

    effect = AREffect(mean_reverting=False)
    effect.fit(y=y, X=None, scale=1.0)

    # Overlapping fh including one in-train and future
    fh = pd.period_range(start="2024-03", periods=3, freq="Q")
    data = effect.transform(X=None, fh=fh)
    _, ix = data
    assert ix.shape[0] == len(fh)
    with seed(numpyro.handlers.seed, 1):
        out = effect.predict(data=data, predicted_effects={})
    assert out.shape == (len(fh), 1)
