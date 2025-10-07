"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from prophetverse._model import inner_model
from prophetverse.distributions import GammaReparametrized
from prophetverse.utils.frame_to_array import series_to_tensor_or_array
from prophetverse.utils.numpyro import CacheMessenger
from .base import BaseEffect
from prophetverse.utils.regex import filter_columns

__all__ = ["LiftExperimentLikelihood"]


class LiftMeasurement(BaseEffect):
    """Wrap effects and applies a likelihood to a certain site.

    This class is supposed to be used as an wrapper around other effects, e.g.:

    ```python
    from prophetverse.effects.linear import LinearEffect
    from prophetverse.effects.measurements import LiftMeasurement
    from prophetverse.effects.target.univariate import NormalTargetLikelihood

    effects = [
        ("linear", LinearEffect(), "A|B"),
        ("target", NormalTargetLikelihood(), None),
    ]

    lift_effect = LiftMeasurement(
        effects=effects,
        measurements=[df_A, df_B, lift_df],
        prior_scale=0.1,
        site_name="obs"
    )
    ```

    Parameters
    ----------
    effects : Tuple[str, BaseEffect, str]
        A tuple with the effect name, the effect instance, and a regex pattern
        or list of columns to be used as input for the effect.
    measurements : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple with three dataframes:
            * The first with A scenario
            * The second with B scenario
            * The third with the lift results
    prior_scale : float
        The scale of the prior distribution for the likelihood.
    likelihood_scale : float
        A multiplier applied to the likelihood to control its contribution.
    site_name : str
        The name to use for the likelihood site in the numpyro model.
    """

    _tags = {
        # Can handle panel data?
        "capability:panel": False,
        # Can handle multiple input feature columns?
        "capability:multivariate_input": True,
        # If no columns are found, should
        # _predict be skipped?
        "requires_X": True,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": False,
        # Is fit() required before calling transform()?
        "requires_fit_before_transform": False,
        # should this effect be applied to `y` (target) or
        # `X` (exogenous variables)?
        "applies_to": "X",
        # Does this effect implement hyperpriors across panel?
        "feature:panel_hyperpriors": False,
        # Should the effect be wrapped with a numpyro.handlers.scope?
        "use_numpyro_scope": False,
    }

    def __init__(
        self,
        effects: Tuple[str, BaseEffect, str],
        measurements: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        prior_scale: float = 0.1,
        likelihood_scale: float = 1.0,
        site_name: str = "obs",
        prefix: str = None,
    ):

        self.effects = effects
        self.measurements = measurements
        self.prior_scale = prior_scale
        self.likelihood_scale = likelihood_scale
        self.site_name = site_name
        self.prefix = prefix

        super().__init__()

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1):
        """Initialize the effect.

        This method is called during `fit()` of the forecasting model.
        It receives the Exogenous variables DataFrame and should be used to initialize
        any necessary parameters or data structures, such as detecting the columns that
        match the regex pattern.


        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale : float, optional
            The scale of the timeseries. For multivariate timeseries, this is
            a dataframe. For univariate, it is a simple float.

        Returns
        -------
        None
        """
        self.all_columns_ = X.columns
        self._training_y = y
        self._effects_and_column_indexes = {}
        self.fitted_effects_ = []
        for effect_name, effect, columns in self.effects:
            effect = effect.clone()
            columns = filter_columns(X, columns)
            if columns is not None:
                columns = list(columns)

            if (
                columns is not None
                and len(columns) == 0
                and effect.get_tag("requires_X", True)
            ):
                raise ValueError(
                    f"Effect '{effect_name}' requires input features but none were matched."
                )

            if columns:
                column_indexes = jnp.array(
                    self.all_columns_.get_indexer(columns), dtype=jnp.int32
                )
                effect_X = X[columns]
            else:
                column_indexes = jnp.array([], dtype=jnp.int32)
                effect_X = None

            effect.fit(y=y, X=effect_X, scale=scale)
            self._effects_and_column_indexes[effect_name] = column_indexes
            self.fitted_effects_.append((effect_name, effect, columns))

        return self

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict[str, Any]:
        """Prepare input data to be passed to numpyro model.

        Returns a dictionary with the data for the lift and for the inner effect.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the exogenous variables for the training
            time indexes, if passed during fit, or for the forecasting time indexes, if
            passed during predict.

        fh : pd.Index
            The forecasting horizon as a pandas Index.

        Returns
        -------
        Dict[str, Any]
            Dictionary with data for the lift and for the inner effect
        """
        scenarious = generate_scenarious(
            X, [self.measurements[0], self.measurements[1]]
        )

        true_data = {}
        for effect_name, effect, columns in self.fitted_effects_:
            effect_input = X[columns] if columns else None
            true_data[effect_name] = effect.transform(
                effect_input,
                fh=fh,
                y=self._training_y,
            )

        all_scenarios = []
        for scenario in scenarious:
            scenario_data = {"data": {}}
            for effect_name, effect, columns in self.fitted_effects_:
                scenario_input = scenario[columns] if columns else None
                scenario_data["data"][effect_name] = effect.transform(
                    scenario_input,
                    fh=fh,
                    y=self._training_y,
                )
            all_scenarios.append(scenario_data)

        lift = self.measurements[2].reindex(fh)

        obs_mask = ~jnp.isnan(series_to_tensor_or_array(lift))
        observed_lift = series_to_tensor_or_array(lift.dropna())

        return {
            "true_data": true_data,
            "scenario_data": all_scenarios,
            "observed_lift": observed_lift,
            "obs_mask": obs_mask,
        }

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries.
        """

        with CacheMessenger(cache_types=["sample", "deterministic"]):
            output_effects = inner_model(
                exogenous_effects={
                    name: effect for name, effect, _ in self.fitted_effects_
                },
                data=data["true_data"],
                predicted_effects=predicted_effects,
            )

            for effect_name, output in output_effects.items():
                predicted_effects[effect_name] = output

            ys = []
            for scenario_data in data["scenario_data"]:

                scenario_outputs = inner_model(
                    exogenous_effects={
                        name: effect for name, effect, _ in self.fitted_effects_
                    },
                    data=scenario_data["data"],
                    predicted_effects=predicted_effects,
                )

                ys.append(scenario_outputs[self.site_name])

            eps = jnp.finfo(ys[0].dtype).eps
            baseline = jnp.clip(ys[0], min=eps)
            counterfactual = jnp.clip(ys[1], min=eps)
            lift = counterfactual / baseline

            site_name = (
                f"{self.prefix}/lift_experiment" if self.prefix else "lift_experiment"
            )
            with numpyro.handlers.scale(scale=self.likelihood_scale):
                distribution = GammaReparametrized(lift, self.prior_scale)

                # Add :ignore so that the model removes this
                # sample when organizing the output dataframe
                numpyro.sample(
                    f"{site_name}:ignore",
                    distribution,
                    obs=data["observed_lift"],
                )

            reference_effect = predicted_effects.get(self.site_name)
            if reference_effect is None:
                first_effect = next(iter(predicted_effects.values()))
                reference_effect = jnp.zeros_like(first_effect)
            else:
                reference_effect = jnp.zeros_like(reference_effect)

            return reference_effect

    def _update_data(self, data, arr):

        for effect_name, effect, _ in self.fitted_effects_:

            indexes = self._effects_and_column_indexes[effect_name]
            if indexes.size == 0:
                continue
            data["true_data"][effect_name] = effect._update_data(
                data["true_data"][effect_name],
                arr[:, indexes],
            )
        return data

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from prophetverse.effects.linear import LinearEffect

        measurements = pd.DataFrame(
            index=pd.date_range(start="2000-01-02", periods=3, freq="D"),
        )
        lift = pd.DataFrame({"lift": [0] * 3}, index=measurements.index)
        return [
            {
                "effects": LinearEffect(),
                "measurements": [measurements, measurements, lift],
                "prior_scale": 0.1,
            }
        ]


def _ensure_dataframe(scenario: Any) -> pd.DataFrame:
    """Convert an input scenario into a pandas DataFrame copy."""

    if isinstance(scenario, pd.DataFrame):
        return scenario.copy()

    if isinstance(scenario, pd.Series):
        return scenario.to_frame().T

    raise TypeError(
        "Each counterfactual scenario must be provided as a pandas DataFrame (or Series)."
    )


def generate_scenarious(X, countefactuals_scenarios):
    """
    Fill missing info in counterfactual scenarios with defaults.

    Receives a dataframe with the original data and a list of
    counterfactual scenarios, that values for specific dates and columns
    according to X format, and missing for unknown dates and columns.

    For example, if X is

    | date       | A | B | C |
    |------------|---|---|---|
    | 2020-01-01 | 1 | 2 | 3 |
    | 2020-01-02 | 4 | 5 | 6 |
    | 2020-01-03 | 7 | 8 | 9 |

    Counterfactuals could be:

    | date       | A | B   | C |
    |------------|---|-----|---|
    | 2020-01-01 | 9 | 2   | 9 |
    | 2020-01-02 | 7 | NaN | 1 |
    | 2020-01-03 | 3 | NaN | 0 |

    And the output would be:

    | date       | A | B | C |
    |------------|---|---|---|
    | 2020-01-01 | 9 | 2 | 9 |
    | 2020-01-02 | 7 | 5 | 1 |
    | 2020-01-03 | 3 | 8 | 0 |

    """

    if countefactuals_scenarios is None:
        return []

    if not isinstance(X, pd.DataFrame):
        raise TypeError("`X` must be a pandas DataFrame.")

    scenarious = []
    for scenario in countefactuals_scenarios:
        scenario_df = _ensure_dataframe(scenario)

        unknown_columns = scenario_df.columns.difference(X.columns)
        if len(unknown_columns) > 0:
            raise ValueError(
                "Counterfactual scenario contains columns not present in `X`: "
                f"{list(unknown_columns)}"
            )

        unknown_index = scenario_df.index.difference(X.index)
        if len(unknown_index) > 0:
            raise ValueError(
                "Counterfactual scenario contains indexes not present in `X`: "
                f"{list(unknown_index)}"
            )

        aligned = scenario_df.reindex(index=X.index, columns=X.columns)
        filled = aligned.fillna(X)

        for column, dtype in X.dtypes.items():
            if column in filled.columns:
                try:
                    filled[column] = filled[column].astype(dtype, copy=False)
                except (TypeError, ValueError):
                    # If casting is not possible (e.g., new float values in an int column),
                    # keep the filled values as-is to avoid data loss.
                    continue

        scenarious.append(filled)

    return scenarious
