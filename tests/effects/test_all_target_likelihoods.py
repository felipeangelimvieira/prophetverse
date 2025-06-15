from skbase.testing.test_all_objects import TestAllObjects, BaseFixtureGenerator
from prophetverse.effects.target.base import BaseTargetEffect
import jax.numpy as jnp
import numpyro
from sktime.utils._testing.series import _make_series


class TargetEffectFixtureGenerator(BaseFixtureGenerator):
    """Fixture for testing all target likelihoods."""

    package_name = "prophetverse.effects.target"
    object_type_filter = BaseTargetEffect


class TestAllTargetEffects(TargetEffectFixtureGenerator, TestAllObjects):

    valid_tags = [
        "capability:panel",
        "capability:multivariate_input",
        "requires_X",
        "applies_to",
        "filter_indexes_with_forecating_horizon_at_transform",
        "requires_fit_before_transform",
        "fitted_named_object_parameters",
        "named_object_parameters",
    ]

    def test_applies_to_tag(self, object_instance):
        assert object_instance.get_tag("applies_to", None) == "y"

    def test_fit_transform_predict_sites(self, object_instance):
        """Test site names and no exceptions"""
        y = _make_series(50).to_frame("value")
        object_instance.fit(y=y, X=None)

        data = object_instance.transform(y, fh=y.index)

        predicted_effects = {
            "trend": jnp.ones(len(y)) * 0.5,
        }
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                predictions = object_instance.predict(data, predicted_effects)

        assert "obs" in trace
        assert trace["obs"]["is_observed"]
        assert all(trace["obs"]["value"].flatten() == y.values.flatten())
        assert "mean" in trace
