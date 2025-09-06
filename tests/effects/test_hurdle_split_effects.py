import pytest
import jax.numpy as jnp

from prophetverse.effects.target.hurdle import HurdleTargetLikelihood


@pytest.mark.smoke
def test_split_no_patterns():
    model = HurdleTargetLikelihood(gate_effect_names=None, gate_effect_only=None)
    predicted_effects = {
        "a": jnp.array([1.0]),
        "b": jnp.array([2.0]),
    }
    gate_only, common, rest = model._split_gate_effects(predicted_effects)
    assert gate_only == {}
    assert common == {}
    assert set(rest.keys()) == {"a", "b"}


@pytest.mark.smoke
def test_split_gate_only_subset():
    model = HurdleTargetLikelihood(
        gate_effect_names=".*",  # everything potentially gate/common
        gate_effect_only="zero__.*",  # only names starting with zero__ go to gate_only
    )
    predicted_effects = {
        "zero__a": jnp.array([1.0]),
        "zero__b": jnp.array([2.0]),
        "c": jnp.array([3.0]),
    }
    gate_only, common, rest = model._split_gate_effects(predicted_effects)
    assert set(gate_only.keys()) == {"zero__a", "zero__b"}
    # remaining (matching gate_effect_names but not gate_effect_only) become common
    assert set(common.keys()) == {"c"}
    assert rest == {}


@pytest.mark.smoke
def test_split_common_only():
    model = HurdleTargetLikelihood(
        gate_effect_names="gate_.*",  # these become common
        gate_effect_only=None,
    )
    predicted_effects = {
        "gate_a": jnp.array([1.0]),
        "gate_b": jnp.array([2.0]),
        "x": jnp.array([3.0]),
    }
    gate_only, common, rest = model._split_gate_effects(predicted_effects)
    assert gate_only == {}
    assert set(common.keys()) == {"gate_a", "gate_b"}
    assert set(rest.keys()) == {"x"}


@pytest.mark.smoke
def test_split_precedence_gate_only_over_common():
    model = HurdleTargetLikelihood(
        gate_effect_names=["gate_only_a", "common_.*"],
        gate_effect_only=["gate_only_a", "gate_only_b"],
    )
    predicted_effects = {
        "gate_only_a": jnp.array([1.0]),  # matches both lists -> should go to gate_only
        "gate_only_b": jnp.array([2.0]),  # only gate_only list
        "common_x": jnp.array([3.0]),  # matches common pattern
        "other": jnp.array([4.0]),  # matches none
    }
    gate_only, common, rest = model._split_gate_effects(predicted_effects)
    assert set(gate_only.keys()) == {"gate_only_a", "gate_only_b"}
    assert set(common.keys()) == {"common_x"}
    assert set(rest.keys()) == {"other"}
