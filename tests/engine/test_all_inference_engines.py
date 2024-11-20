import numpyro
from skbase.testing.test_all_objects import BaseFixtureGenerator, QuickTester

from prophetverse.engine.base import BaseInferenceEngine


def _model(obs):

    mean = numpyro.sample("mean", numpyro.distributions.Normal(0, 1))
    return numpyro.sample("y", numpyro.distributions.Normal(mean, 1), obs=obs)


class InferenceEngineFixtureGenerator(BaseFixtureGenerator):
    object_type_filter = BaseInferenceEngine
    exclude_objects = ["InferenceEngine"]
    package_name = "prophetverse.engine"


class TestAllInferenceEngines(InferenceEngineFixtureGenerator, QuickTester):

    def test_inference_converges(self, object_instance):
        import jax.numpy as jnp
        import numpy as np

        obs = jnp.array(np.random.normal(0, 1, 100))
        object_instance.infer(_model, obs=obs)

        assert isinstance(object_instance.posterior_samples_, dict)
        assert "mean" in object_instance.posterior_samples_
        assert jnp.isfinite(object_instance.posterior_samples_["mean"].mean().item())
