from skbase.testing.test_all_objects import TestAllObjects
from prophetverse.effects.base import BaseEffect


class TestAllEffects(TestAllObjects):

    package_name = "prophetverse.effects"
    valid_tags = [
        "capability:panel",
        "capability:multivariate_input",
        "requires_X",
        "filter_indexes_with_forecating_horizon_at_transform",
        "requires_fit_before_transform",
        "fitted_named_object_parameters",
        "named_object_parameters",
    ]

    object_type_filter = BaseEffect
