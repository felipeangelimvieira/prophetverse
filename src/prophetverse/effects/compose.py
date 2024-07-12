"""Composition of effects (Effects that wrap other effects)."""

from typing import Any, Dict, List

import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd

from prophetverse.utils.frame_to_array import series_to_tensor_or_array

from .base import BaseEffect, Stage


class LiftExperimentLikelihood(BaseEffect):
    """Wrap an effect and applies a normal likelihood to its output.

    This class uses an input as a reference for the effect, and applies a normal
    likelihood to the output of the effect.

    Parameters
    ----------
    effect : BaseEffect
        The effect to wrap.
    lift_test_results : pd.DataFrame
        A dataframe with the lift test results. Should be in sktime format, and must
        have the same index as the input data.
    prior_scale : float
        The scale of the prior distribution for the likelihood.


    """

    _tags = {"skip_apply_if_no_match": False, "supports_multivariate": False}

    def __init__(
        self,
        effect: BaseEffect,
        lift_test_results: pd.DataFrame,
        prior_scale: float,
    ):

        self.effect = effect
        self.lift_test_results = lift_test_results
        self.prior_scale = prior_scale

        assert self.prior_scale > 0, "prior_scale must be greater than 0"

        super().__init__(id=self.effect.id, regex=None)

    def initialize(self, X: pd.DataFrame, scale: float = 1):
        """Initialize this effect and its wrapped effect.

        Parameters
        ----------
        X : DataFrame
            Dataframe of exogenous data.
        scale : float
            The scale of the timeseries. This is used to normalize the lift effect.
        """
        self.effect.initialize(X)
        self.timeseries_scale = scale
        super().initialize(X)

    @property
    def input_feature_column_names(self) -> List[str]:
        """Return the input feature columns names."""
        return self.effect._input_feature_column_names

    def _prepare_input_data(
        self, X: pd.DataFrame, stage: Stage = Stage.TRAIN
    ) -> Dict[str, Any]:
        """Prepare the input data for the effect, and the custom likelihood.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with exogenous variables.
        stage : Stage, optional
            which stage is being executed, by default Stage.TRAIN.
            Used to determine if the likelihood should be applied.

        Returns
        -------
        Dict[str, Any]
            The dictionary of data passed to _apply and the likelihood.
        """
        data_dict = self.effect._prepare_input_data(X, stage)

        if stage == Stage.PREDICT:
            data_dict["observed_lift"] = None
            return data_dict

        X_lift = self.lift_test_results.loc[X.index]
        lift_array = series_to_tensor_or_array(X_lift)
        data_dict["observed_lift"] = lift_array / self.timeseries_scale
        return data_dict

    def apply(
        self,
        trend: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Apply the effect and the custom likelihood.

        Parameters
        ----------
        trend : jnp.ndarray
            The trend component.
        observed_lift : jnp.ndarray
            The observed lift to apply the likelihood to.

        Returns
        -------
        jnp.ndarray
            The effect applied to the input data.
        """
        observed_lift = kwargs.pop("observed_lift")

        x = self.effect.apply(trend, **kwargs)

        self.sample(
            f"lift_experiment_{self.effect.id}",
            dist.Normal(x, self.prior_scale),
            obs=observed_lift,
        )

        return x
