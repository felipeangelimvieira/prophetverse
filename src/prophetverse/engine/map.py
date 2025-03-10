"""Numpyro inference engines for prophet models.

The classes in this module take a model, the data and perform inference using Numpyro.
"""

from typing import Optional

import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_mean
from numpyro.infer.svi import SVIRunResult

from prophetverse.engine.base import BaseInferenceEngine
from prophetverse.engine.optimizer.optimizer import (
    AdamOptimizer,
    BaseOptimizer,
    LBFGSSolver,
)
from prophetverse.utils.deprecation import deprecation_warning

_DEFAULT_PREDICT_NUM_SAMPLES = 1000
DEFAULT_PROGRESS_BAR = False


class MAPInferenceEngine(BaseInferenceEngine):
    """
    Maximum a Posteriori (MAP) Inference Engine.

    This class performs MAP inference using Stochastic Variational Inference (SVI)
    with AutoDelta guide. It provides methods for inference and prediction.

    Parameters
    ----------
    optimizer : Optional[BaseOptimizer]
        The optimizer to be used for inference. If not provided, the default is
        LBFGSSolver.
    num_steps : int, optional
        The number of steps to run the optimizer. Default is 10000.
    num_samples : int, optional
        The number of samples to generate during prediction.
        Default is _DEFAULT_PREDICT_NUM_SAMPLES.
    rng_key : optional
        The random number generator key.
    progress_bar : bool, optional
        Whether to display a progress bar during inference. Default is DEFAULT_PROGRESS_BAR.
    stable_update : bool, optional
        Whether to use stable update during inference. Default is False.
    forward_mode_differentiation : bool, optional
        Whether to use forward mode differentiation. Default is False.
    init_loc_fn : optional
        The function to initialize the location parameter. If not provided, the default is init_to_mean.

    """

    _tags = {
        "inference_method": "map",
    }

    def __init__(
        self,
        optimizer: Optional[BaseOptimizer] = None,
        num_steps=10_000,
        num_samples=_DEFAULT_PREDICT_NUM_SAMPLES,
        rng_key=None,
        progress_bar: bool = DEFAULT_PROGRESS_BAR,
        stable_update=False,
        forward_mode_differentiation=False,
        init_loc_fn=None,
    ):

        self.optimizer = optimizer
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.progress_bar = progress_bar
        self.stable_update = stable_update
        self.forward_mode_differentiation = forward_mode_differentiation
        self.init_loc_fn = init_loc_fn
        super().__init__(rng_key)

        if optimizer is None:
            optimizer = LBFGSSolver()

        self._optimizer = optimizer

        self._init_loc_fn = init_loc_fn
        if init_loc_fn is None:
            self._init_loc_fn = init_to_mean()

        self._num_steps = num_steps

        if self._optimizer.get_tag("is_solver", False):  # type: ignore[union-attr]
            # If solver, there's a single "solver step". For compatibility,
            # we set num_steps to 1 and max_iter to the original num_steps.

            self._optimizer = self._optimizer.set_max_iter(  # type: ignore[union-attr]
                self._num_steps
            )
            self._num_steps = 1

    def _infer(self, **kwargs):
        """
        Perform MAP inference.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        self
            The updated MAPInferenceEngine object.
        """
        self.guide_ = AutoDelta(self.model_, init_loc_fn=self._init_loc_fn)

        def get_result(
            rng_key,
            model,
            guide,
            optimizer,
            num_steps,
            progress_bar,
            stable_update,
            forward_mode_differentiation,
            **kwargs,
        ) -> SVIRunResult:
            svi_ = SVI(
                model,
                guide,
                optimizer,
                loss=Trace_ELBO(),
            )
            return svi_.run(
                rng_key=rng_key,
                progress_bar=progress_bar,
                stable_update=stable_update,
                num_steps=num_steps,
                forward_mode_differentiation=forward_mode_differentiation,
                **kwargs,
            )

        self.run_results_: SVIRunResult = get_result(
            self._rng_key,
            self.model_,
            self.guide_,
            self._optimizer.create_optimizer(),
            self._num_steps,
            stable_update=self.stable_update,
            progress_bar=self.progress_bar,
            forward_mode_differentiation=self.forward_mode_differentiation,
            **kwargs,
        )

        self.raise_error_if_nan_loss(self.run_results_)

        self.posterior_samples_ = self.guide_.sample_posterior(
            self._rng_key, params=self.run_results_.params, **kwargs
        )
        return self

    def raise_error_if_nan_loss(self, run_results: SVIRunResult):
        """
        Raise an error if the loss is NaN.

        Parameters
        ----------
        run_results : SVIRunResult
            The result of the SVI run.

        Raises
        ------
        MAPInferenceEngineError
            If the last loss is NaN.
        """
        losses = run_results.losses
        if jnp.isnan(losses)[-1]:
            msg = "NaN losses in MAPInferenceEngine."
            msg += " Try decreasing the learning rate or changing the model specs."
            msg += " If the problem persists, please open an issue at"
            msg += " https://github.com/felipeangelimvieira/prophetverse"
            raise MAPInferenceEngineError(msg)

    def _predict(self, **kwargs):
        """
        Generate predictions using the trained model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        dict
            The predicted samples generated by the model.
        """
        predictive = numpyro.infer.Predictive(
            self.model_,
            params=self.run_results_.params,
            guide=self.guide_,
            # posterior_samples=self.posterior_samples_,
            num_samples=self.num_samples,
        )
        self.samples_ = predictive(rng_key=self._rng_key, **kwargs)
        return self.samples_

    @classmethod
    def get_test_params(*args, **kwargs):
        """Return test params for unit testing."""
        return [
            {
                "optimizer": LBFGSSolver(),
                "num_steps": 100,
            },
            {
                "optimizer": AdamOptimizer(),
                "num_steps": 100,
            },
        ]


class MAPInferenceEngineError(Exception):
    """Exception raised for NaN losses in MAPInferenceEngine."""

    def __init__(self, message="NaN losses in MAPInferenceEngine"):
        self.message = message
        super().__init__(self.message)
