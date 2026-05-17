"""Tests that Prophetverse MAP inference is insensitive to PYTHONHASHSEED.

The results of fitting a Prophetverse model must be identical regardless of
the ``PYTHONHASHSEED`` environment variable set for the Python process.

These tests use a subprocess-based approach because ``PYTHONHASHSEED`` is
read only at interpreter start-up and cannot be changed at runtime.
"""

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helper: run a small prophetverse fit in a fresh subprocess and return the
# first prediction value as a string so we can compare across runs.
# ---------------------------------------------------------------------------

_FIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    sys.path.insert(0, '{src_path}')

    import numpy as np
    import pandas as pd
    import jax

    rng = np.random.RandomState(42)
    dates = pd.period_range("2020", periods=36, freq="Q")
    y = pd.DataFrame({{"y": rng.randn(36).cumsum() + 100}}, index=dates)

    from prophetverse.sktime import Prophetverse
    from prophetverse.engine import MAPInferenceEngine
    from prophetverse.engine.optimizer import LBFGSSolver
    from prophetverse.effects.trend import PiecewiseLinearTrend

    trend = PiecewiseLinearTrend(
        changepoint_interval=8,
        changepoint_range=-2,
        changepoint_prior_scale=0.001,
    )
    model = Prophetverse(
        trend=trend,
        inference_engine=MAPInferenceEngine(
            optimizer=LBFGSSolver(memory_size=10, max_linesearch_steps=30),
            num_steps=100,
        ),
    )
    model.fit(y=y)
    pred = model.predict(fh=y.index)
    # Print the first 5 prediction values as CSV so the test can parse them
    vals = pred.values.flatten()[:5]
    print(",".join(f"{{v:.6f}}" for v in vals))
    """
)


def _run_with_hashseed(seed: int, src_path: str) -> list[float]:
    """Run the fit script with PYTHONHASHSEED=*seed* and return predictions."""
    env = {"PYTHONHASHSEED": str(seed)}
    # Inherit the PATH/PYTHONPATH from the current process
    import os

    full_env = {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-c", _FIT_SCRIPT.format(src_path=src_path)],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (PYTHONHASHSEED={seed}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    # Parse the comma-separated float values
    try:
        values = [float(v) for v in result.stdout.strip().split(",")]
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse output (PYTHONHASHSEED={seed}): {result.stdout!r}"
        ) from exc
    return values


@pytest.fixture(scope="module")
def src_path():
    """Return the path to the prophetverse source tree."""
    import prophetverse

    import os

    return os.path.dirname(os.path.dirname(prophetverse.__file__))


@pytest.mark.ci
def test_map_lbfgs_insensitive_to_pythonhashseed(src_path):
    """MAP inference with LBFGS must not be affected by PYTHONHASHSEED.

    This is a regression test for the bug where numpyro's ``Trace_ELBO``
    iterated over an unordered Python ``set`` of site names, making the
    floating-point summation order hash-randomised.  Different PYTHONHASHSEED
    values caused different LBFGS convergence behaviour and ultimately
    different MAP estimates.
    """
    seed_values = [0, 1, 42]
    predictions = [_run_with_hashseed(s, src_path) for s in seed_values]

    reference = predictions[0]
    for seed_val, preds in zip(seed_values[1:], predictions[1:]):
        for i, (ref_val, pred_val) in enumerate(zip(reference, preds)):
            assert abs(ref_val - pred_val) < 1e-4, (
                f"Prediction[{i}] differs between PYTHONHASHSEED=0 "
                f"({ref_val:.6f}) and PYTHONHASHSEED={seed_val} "
                f"({pred_val:.6f}).  Prophetverse MAP inference must be "
                f"insensitive to PYTHONHASHSEED."
            )
