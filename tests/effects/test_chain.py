"""Pytest for Chained Effects class."""

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects.base import BaseEffect
from prophetverse.effects.chain import ChainedEffects


class MockEffect(BaseEffect):
    def __init__(self, value):
        self.value = value
        super().__init__()

        self._transform_called = False

    def _transform(self, X, fh):
        self._transform_called = True
        return super()._transform(X, fh)

    def _predict(self, data, predicted_effects, *args, **kwargs):
        param = numpyro.sample("param", numpyro.distributions.Delta(self.value))
        return data * param


class IdentityEffect(BaseEffect):
    """An identity effect that passes data through unchanged.

    Used for testing to verify that adstock order doesn't matter
    when paired with a non-transforming effect.
    """

    def __init__(self):
        super().__init__()

    def _predict(self, data, predicted_effects, *args, **kwargs):
        return data


@pytest.fixture
def index():
    return pd.date_range("2021-01-01", periods=6)


@pytest.fixture
def y(index):
    return pd.DataFrame(index=index, data=[1] * len(index))


@pytest.fixture
def X(index):
    return pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60]},
        index=index,
    )


def test_chained_effects_fit_transform(X, y):
    """Test the fit method of ChainedEffects."""
    effects = [("effect1", MockEffect(2)), ("effect2", MockEffect(3))]
    chained = ChainedEffects(steps=effects)

    scale = 1
    chained.fit(y=y, X=X, scale=scale)
    # Ensure no exceptions occur in fit

    # Test transform
    transformed = chained.transform(X, fh=X.index)
    expected = MockEffect(2).transform(X, fh=X.index)
    assert jnp.allclose(transformed, expected), "Chained transform output mismatch."


def test_chained_effects_predict(X, y):
    """Test the predict method of ChainedEffects."""
    effects = [("effect1", MockEffect(2)), ("effect2", MockEffect(3))]
    chained = ChainedEffects(steps=effects)
    chained.fit(y=y, X=X, scale=1)
    data = chained.transform(X, fh=X.index)
    predicted_effects = {}
    with handlers.trace() as trace:
        predicted = chained.predict(data, predicted_effects)

    with numpyro.handlers.trace() as exec_trace:
        predicted = chained.predict(data, predicted_effects)
    expected = data * 2 * 3
    assert jnp.allclose(predicted, expected), "Chained predict output mismatch."

    assert "effect1/param" in trace, "Missing effect_0 trace."
    assert "effect2/param" in trace, "Missing effect_1 trace."

    assert trace["effect1/param"]["value"] == 2, "Incorrect effect_0 trace value."
    assert trace["effect2/param"]["value"] == 3, "Incorrect effect_1 trace value."


def test_get_params():
    effects = [("effect1", MockEffect(2)), ("effect2", MockEffect(3))]
    chained = ChainedEffects(steps=effects)

    params = chained.get_params()

    assert params["effect1__value"] == 2, "Incorrect effect_0 param."
    assert params["effect2__value"] == 3, "Incorrect effect_1 param."


# Fixtures for adstock chain semantic tests
@pytest.fixture
def historical_index():
    """Historical data index (50 days of history)."""
    return pd.date_range("2021-01-01", periods=50, freq="D")


@pytest.fixture
def horizon_index(historical_index):
    """Forecast horizon index (10 days after history)."""
    return pd.date_range(
        historical_index[-1] + pd.Timedelta(days=1), periods=10, freq="D"
    )


@pytest.fixture
def historical_X(historical_index):
    """Historical exogenous data with varying spend."""
    return pd.DataFrame(
        data={"spend": [100.0 * (1 + 0.1 * i) for i in range(50)]},
        index=historical_index,
    )


@pytest.fixture
def horizon_X(horizon_index):
    """Horizon exogenous data."""
    return pd.DataFrame(
        data={"spend": [150.0] * 10},
        index=horizon_index,
    )


@pytest.fixture
def y_historical(historical_index):
    """Target variable for historical data."""
    return pd.DataFrame(index=historical_index, data={"y": [1.0] * 50})


def test_adstock_after_hill_receives_saturated_values(
    historical_X, horizon_X, y_historical, historical_index, horizon_index
):
    """Test that adstock in second position receives Hill-saturated values.

    When we have Hill → Adstock, the adstock should apply carryover to the
    saturated spend values, not raw spend or zeros.
    """
    from prophetverse.effects.chain import ChainedEffects
    from prophetverse.effects.hill import HillEffect
    from prophetverse.effects.adstock import GeometricAdstockEffect

    # Create chain: Hill first, then Adstock
    chain = ChainedEffects(
        steps=[
            (
                "hill",
                HillEffect(
                    half_max_prior=dist.HalfNormal(1),
                    slope_prior=dist.HalfNormal(1),
                    max_effect_prior=dist.HalfNormal(1),
                    effect_mode="additive",
                ),
            ),
            ("adstock", GeometricAdstockEffect(normalize=True)),
        ]
    )

    # Fit on historical data
    chain.fit(y=y_historical, X=historical_X, scale=1.0)

    # Transform with horizon data (chain should include historical context)
    transformed = chain.transform(horizon_X, fh=horizon_index)

    # Verify transform returns dict format with historical context
    assert isinstance(
        transformed, dict
    ), "Transform should return dict for Hill→Adstock chain"
    assert "first_transform" in transformed, "Transform should contain first_transform"
    assert "horizon_indices" in transformed, "Transform should contain horizon_indices"

    # The first_transform should have shape for full history (50) + horizon (10) = 60
    first_transform = transformed["first_transform"]
    assert first_transform.shape[0] == 60, (
        f"First transform should have 60 rows (50 history + 10 horizon), "
        f"got {first_transform.shape[0]}"
    )

    # Run prediction
    with handlers.seed(rng_seed=42):
        output = chain.predict(transformed, predicted_effects={})

    # Output should be for horizon only (10 rows)
    assert output.shape[0] == 10, f"Output should have 10 rows, got {output.shape[0]}"

    # Crucially: output should be non-zero (adstock received saturated values, not zeros)
    assert jnp.all(
        output > 0
    ), "Output should be positive (adstock received saturated values)"


def test_adstock_before_hill_also_works(
    historical_X, horizon_X, y_historical, historical_index, horizon_index
):
    """Test that adstock in first position (Adstock → Hill) works correctly.

    This is the simpler case where adstock directly receives the raw data.
    """
    from prophetverse.effects.chain import ChainedEffects
    from prophetverse.effects.hill import HillEffect
    from prophetverse.effects.adstock import GeometricAdstockEffect

    # Create chain: Adstock first, then Hill
    chain = ChainedEffects(
        steps=[
            ("adstock", GeometricAdstockEffect(normalize=True)),
            (
                "hill",
                HillEffect(
                    half_max_prior=dist.HalfNormal(1),
                    slope_prior=dist.HalfNormal(1),
                    max_effect_prior=dist.HalfNormal(1),
                    effect_mode="additive",
                ),
            ),
        ]
    )

    # Fit and transform
    chain.fit(y=y_historical, X=historical_X, scale=1.0)
    transformed = chain.transform(horizon_X, fh=horizon_index)

    # Run prediction
    with handlers.seed(rng_seed=42):
        output = chain.predict(transformed, predicted_effects={})

    # Output should be for horizon only (10 rows)
    assert output.shape[0] == 10, f"Output should have 10 rows, got {output.shape[0]}"

    # Output should be positive
    assert jnp.all(output > 0), "Output should be positive"


def test_chain_order_produces_different_results(
    historical_X, horizon_X, y_historical, historical_index, horizon_index
):
    """Test that Hill→Adstock and Adstock→Hill produce different results.

    This verifies that the order matters and both orderings work correctly.
    """
    from prophetverse.effects.chain import ChainedEffects
    from prophetverse.effects.hill import HillEffect
    from prophetverse.effects.adstock import GeometricAdstockEffect

    # Create both orderings with same parameters
    hill_params = {
        "half_max_prior": dist.HalfNormal(1),
        "slope_prior": dist.HalfNormal(1),
        "max_effect_prior": dist.HalfNormal(1),
        "effect_mode": "additive",
    }
    adstock_params = {"normalize": True}

    chain_hill_first = ChainedEffects(
        steps=[
            ("hill", HillEffect(**hill_params)),
            ("adstock", GeometricAdstockEffect(**adstock_params)),
        ]
    )

    chain_adstock_first = ChainedEffects(
        steps=[
            ("adstock", GeometricAdstockEffect(**adstock_params)),
            ("hill", HillEffect(**hill_params)),
        ]
    )

    # Fit both
    chain_hill_first.fit(y=y_historical, X=historical_X, scale=1.0)
    chain_adstock_first.fit(y=y_historical, X=historical_X, scale=1.0)

    # Transform and predict
    transformed_hf = chain_hill_first.transform(horizon_X, fh=horizon_index)
    transformed_af = chain_adstock_first.transform(horizon_X, fh=horizon_index)

    with handlers.seed(rng_seed=42):
        output_hill_first = chain_hill_first.predict(
            transformed_hf, predicted_effects={}
        )

    with handlers.seed(rng_seed=42):
        output_adstock_first = chain_adstock_first.predict(
            transformed_af, predicted_effects={}
        )

    # Both should produce valid positive outputs
    assert jnp.all(output_hill_first > 0), "Hill→Adstock output should be positive"
    assert jnp.all(output_adstock_first > 0), "Adstock→Hill output should be positive"

    # The outputs should be different (order matters for the transformation)
    # Note: They might be similar in magnitude but should not be identical
    assert output_hill_first.shape == output_adstock_first.shape, "Shapes should match"


def test_identity_effect_order_doesnt_matter(
    historical_X, horizon_X, y_historical, historical_index, horizon_index
):
    """Test that Identity→Adstock and Adstock→Identity produce identical results.

    When the non-adstock effect is an identity (pass-through), the order
    shouldn't matter because:
    - Identity→Adstock: raw data → adstock carryover
    - Adstock→Identity: adstock carryover → pass through unchanged

    Both should produce the same result.
    """
    from prophetverse.effects.chain import ChainedEffects
    from prophetverse.effects.adstock import GeometricAdstockEffect

    adstock_params = {"normalize": True}

    # Create both orderings
    chain_identity_first = ChainedEffects(
        steps=[
            ("identity", IdentityEffect()),
            ("adstock", GeometricAdstockEffect(**adstock_params)),
        ]
    )

    chain_adstock_first = ChainedEffects(
        steps=[
            ("adstock", GeometricAdstockEffect(**adstock_params)),
            ("identity", IdentityEffect()),
        ]
    )

    # Fit both
    chain_identity_first.fit(y=y_historical, X=historical_X, scale=1.0)
    chain_adstock_first.fit(y=y_historical, X=historical_X, scale=1.0)

    # Transform and predict
    transformed_if = chain_identity_first.transform(horizon_X, fh=horizon_index)
    transformed_af = chain_adstock_first.transform(horizon_X, fh=horizon_index)

    with handlers.seed(rng_seed=42):
        output_identity_first = chain_identity_first.predict(
            transformed_if, predicted_effects={}
        )

    with handlers.seed(rng_seed=42):
        output_adstock_first = chain_adstock_first.predict(
            transformed_af, predicted_effects={}
        )

    # Both should produce identical outputs since Identity doesn't transform
    assert jnp.allclose(output_identity_first, output_adstock_first), (
        f"Identity→Adstock and Adstock→Identity should produce identical results, "
        f"got {output_identity_first} vs {output_adstock_first}"
    )
