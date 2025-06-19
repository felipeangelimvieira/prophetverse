from skbase.testing.test_all_objects import TestAllObjects
from prophetverse.effects.base import BaseEffect

from sktime.utils._testing.series import _make_series
import pandas as pd
import pytest

import jax.numpy as jnp
import numpyro
from prophetverse.utils.frame_to_array import series_to_tensor_or_array

RAND_SEED = 42


def _make_panel(n_instances, n_timepoints, n_columns=1, random_state=None):

    data = {}
    for i in range(n_instances):
        data[f"instance_{i}"] = _make_series(
            n_timepoints=n_timepoints, n_columns=n_columns, random_state=random_state
        )
    return pd.concat(data, axis=0)


class Scenario:
    """A simple scenario object for demonstration."""

    def __init__(self, y, X):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        self.y = y
        self.X = X
        self.fh = y.index.get_level_values(-1).unique()[1:]

    @property
    def trend(self):
        n_timepoints = len(self.fh)
        if self.y.index.nlevels > 1:
            n_series = self.y.index.droplevel(-1).nunique()

            return jnp.ones((n_series, n_timepoints, 1))
        return jnp.ones((n_timepoints, 1))


class TestAllEffects(TestAllObjects):

    package_name = "prophetverse.effects"
    valid_tags = [
        "hierarchical_prophet_compliant",
        "capability:panel",
        "capability:multivariate_input",
        "requires_X",
        "applies_to",
        "filter_indexes_with_forecating_horizon_at_transform",
        "requires_fit_before_transform",
        "fitted_named_object_parameters",
        "named_object_parameters",
        "feature:panel_hyperpriors",
    ]

    object_type_filter = BaseEffect

    fixture_sequence = ["object_class", "object_instance", "scenario"]

    def _generate_scenario(self, test_name, **kwargs):
        """
        Generates scenarios for testing.

        It can optionally use kwargs to access earlier fixtures, e.g.,
        object_instance = kwargs.get("object_instance")
        to create object-specific scenarios.
        """

        scenarios = [
            Scenario(
                y=_make_series(n_timepoints=50, n_columns=1, random_state=RAND_SEED),
                X=_make_series(n_columns=2, n_timepoints=50, random_state=RAND_SEED),
            ),
            Scenario(
                y=_make_series(n_timepoints=50, n_columns=1, random_state=RAND_SEED),
                X=_make_series(n_columns=1, n_timepoints=50, random_state=RAND_SEED),
            ),
            Scenario(
                y=_make_series(n_timepoints=50, n_columns=1, random_state=RAND_SEED),
                X=None,
            ),
            Scenario(
                y=_make_panel(
                    n_instances=3,
                    n_timepoints=50,
                    n_columns=1,
                    random_state=RAND_SEED,
                ),
                X=_make_panel(
                    n_instances=3, n_timepoints=50, n_columns=2, random_state=RAND_SEED
                ),
            ),
            Scenario(
                y=_make_panel(
                    n_instances=3, n_timepoints=50, n_columns=1, random_state=RAND_SEED
                ),
                X=None,
            ),
        ]
        scenario_names = [
            "single_series_two_features",
            "single_series_one_feature",
            "single_series_no_features",
            "panel_two_features",
            "panel_no_features",
        ]

        return scenarios, scenario_names

    def test_object_behavior_with_scenario(self, object_instance, scenario):
        """
        A custom test that uses the object and scenario fixtures.

        This test will be executed for each combination of object instance
        and scenario generated.
        """
        # This is a placeholder for your actual test logic.
        # For example:
        # result = object_instance.predict(scenario.data)
        # assert result == scenario.expected_output

        y = scenario.y
        X = scenario.X

        requires_x = object_instance.get_tag("requires_X")
        if requires_x and X is None:
            pytest.skip("This effect requires X, but X is None.")

        if X is None:
            X = pd.DataFrame(index=y.index)

        object_instance.fit(y=y, X=X, scale=2)

        applies_to = object_instance.get_tag("applies_to")
        if applies_to == "y":
            data = object_instance.transform(y, fh=scenario.fh)

        elif applies_to == "X":
            data = object_instance.transform(X, fh=scenario.fh)

        assert self._validate_transform_output(data)

        predicted_effects = {"trend": scenario.trend}

        with numpyro.handlers.seed(rng_seed=0):
            out = object_instance.predict(
                data=data, predicted_effects=predicted_effects
            )

        assert (
            out.shape == scenario.trend.shape
        ), f"Expected output shape {scenario.trend.shape}, but got {out.shape}"

    def _validate_transform_output(self, obj):

        is_valid_dict = False
        is_valid_array = False
        is_valid_tuple = False

        if obj is None:
            return True

        if isinstance(obj, dict):
            is_valid_dict = self._validate_transform_output(obj.get("data"))

        if isinstance(obj, jnp.ndarray):
            is_valid_array = obj.ndim == 2 or obj.ndim == 3

        if isinstance(obj, tuple):
            is_valid_tuple = isinstance(obj[0], jnp.ndarray)

        is_valid = is_valid_array or is_valid_dict or is_valid_tuple
        if isinstance(obj, list):
            is_valid = all(self._validate_transform_output(o) for o in obj)
        return is_valid
