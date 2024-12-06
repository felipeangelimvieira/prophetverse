import numpyro
from skbase.testing.test_all_objects import BaseFixtureGenerator, QuickTester

from prophetverse.engine.optimizer import BaseOptimizer


class OptimizerFixtureGenerator(BaseFixtureGenerator):

    object_type_filter = BaseOptimizer
    package_name = "prophetverse.engine"


class TestAllOptimizers(OptimizerFixtureGenerator, QuickTester):

    def test_optimizer_output_class(self, object_instance):
        assert isinstance(
            object_instance.create_optimizer(), numpyro.optim._NumPyroOptim
        )
