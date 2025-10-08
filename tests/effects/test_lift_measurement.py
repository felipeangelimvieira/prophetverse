import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers

from prophetverse.effects.linear import LinearEffect
from prophetverse.effects.measurements import LiftMeasurement
from prophetverse.effects.trend.flat import FlatTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.engine.optimizer import AdamOptimizer
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.utils.numpyro import CacheMessenger


@pytest.fixture
def base_exog():
    index = pd.date_range("2021-01-01", periods=3)
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
        },
        index=index,
    )


@pytest.fixture
def measurements(base_exog):
    a_scenario = base_exog.copy()
    b_scenario = base_exog.copy()
    lift_df = pd.DataFrame({"lift": jnp.ones(len(base_exog))}, index=base_exog.index)
    return a_scenario, b_scenario, lift_df


@pytest.fixture
def y(base_exog):
    return pd.DataFrame({"target": [10.0, 10.0, 10.0]}, index=base_exog.index)


@pytest.fixture
def lift_measurement_effect(measurements):
    effects = [
        (
            "linear_a",
            LinearEffect(prior=dist.Delta(1.0), effect_mode="additive"),
            "^A$",
        ),
        (
            "linear_b",
            LinearEffect(prior=dist.Delta(1.0), effect_mode="additive"),
            "^B$",
        ),
    ]
    return LiftMeasurement(
        effects=effects,
        measurements=measurements,
        prior_scale=0.1,
        site_name="linear_a",
    )


def test_lift_measurement_transform_populates_all_scenarios(
    lift_measurement_effect, base_exog, y
):
    fh = base_exog.index
    lift_measurement_effect.fit(y=y, X=base_exog, scale=1.0)
    transformed = lift_measurement_effect.transform(base_exog, fh=fh)

    assert set(transformed["true_data"].keys()) == {"linear_a", "linear_b"}
    assert len(transformed["scenario_data"]) == 2
    for scenario in transformed["scenario_data"]:
        assert set(scenario["data"].keys()) == {"linear_a", "linear_b"}


def test_lift_measurement_predict_identical_scenarios_has_unit_lift(
    lift_measurement_effect, base_exog, y
):
    fh = base_exog.index

    lift_measurement_effect.fit(y=y, X=base_exog, scale=1.0)
    data = lift_measurement_effect.transform(base_exog, fh=fh)

    exec_trace = handlers.trace(lift_measurement_effect.predict).get_trace(
        data=data,
        predicted_effects={},
    )

    assert "lift_experiment:ignore" in exec_trace
    gamma_dist = exec_trace["lift_experiment:ignore"]["fn"]
    assert jnp.allclose(gamma_dist.loc, jnp.ones_like(gamma_dist.loc))


def test_lift_measurement_with_prophetverse_dummy_dataset():
    index = pd.date_range("2022-01-01", periods=6, freq="D")
    y = pd.DataFrame({"target": jnp.linspace(10.0, 15.0, num=6)}, index=index)
    X = pd.DataFrame(
        {
            "A": jnp.linspace(1.0, 6.0, num=6),
            "B": jnp.linspace(2.0, 7.0, num=6),
        },
        index=index,
    )

    lift_df = pd.DataFrame({"lift": jnp.ones(len(index))}, index=index)

    measurements = (
        X.iloc[:4].copy(),
        X.iloc[:4].copy(),
        lift_df.iloc[:4].copy(),
    )

    future_X = X.iloc[4:]
    future_measurements = (
        future_X.copy(),
        future_X.copy(),
        pd.DataFrame({"lift": jnp.ones(len(future_X))}, index=future_X.index),
    )

    lift_effect = LiftMeasurement(
        effects=[
            (
                "linear_a",
                LinearEffect(prior=dist.Delta(1.0), effect_mode="additive"),
                "^A$",
            ),
            (
                "linear_b",
                LinearEffect(prior=dist.Delta(1.0), effect_mode="additive"),
                "^B$",
            ),
        ],
        measurements=measurements,
        prior_scale=0.1,
        likelihood_scale=1.0,
        site_name="linear_a",
    )

    forecaster = Prophetverse(
        trend=FlatTrend(),
        exogenous_effects=[
            (
                "lift_measurement",
                lift_effect,
                "^(A|B)$",
            )
        ],
        inference_engine=MAPInferenceEngine(
            optimizer=AdamOptimizer(step_size=0.01), num_steps=1, num_samples=1
        ),
    )

    train_y, train_X = y.iloc[:4], X.iloc[:4]
    forecaster.fit(train_y, train_X)

    fh = list(range(1, len(future_X) + 1))
    for name, effect, _ in forecaster.all_effects_:
        if name == "lift_measurement":
            effect.measurements = future_measurements
            break
    predictions = forecaster.predict(X=future_X, fh=fh)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (len(fh), 1)

    components = forecaster.predict_components(fh=fh, X=future_X)
    assert isinstance(components, pd.DataFrame)

    def _normalize_component_name(name):
        if isinstance(name, tuple):
            return "/".join(str(part) for part in name)
        return str(name)

    component_names = {_normalize_component_name(col) for col in components.columns}

    inner_effect_names = {f"{effect_name}" for effect_name, _, _ in lift_effect.effects}

    assert inner_effect_names.issubset(component_names)

    fh_index = components.index.get_level_values(-1)
    with pytest.raises(KeyError):
        _ = components["non_existent_component"]

    # Ensure wrapped effects remain visible to other effects via predicted_effects
    with CacheMessenger(), handlers.trace() as tr:
        forecaster.predict_components(fh=fh, X=future_X)

    predicted_effects = {}
    for site_name, site in tr.items():
        if site["type"] == "deterministic":
            normalized_name = _normalize_component_name(site_name)
            predicted_effects[normalized_name] = site["value"]

    for effect_name in inner_effect_names:
        assert effect_name in predicted_effects
        assert predicted_effects[effect_name].shape[-1] == 1
