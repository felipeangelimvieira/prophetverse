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
from prophetverse.engine.optimizer import (
    AdamOptimizer,
    BaseOptimizer,
    _OptimizerFromCallable,
)
from prophetverse.utils.deprecation import deprecation_warning

_DEFAULT_PREDICT_NUM_SAMPLES = 1000
DEFAULT_PROGRESS_BAR = True


class MAPInferenceEngine(BaseInferenceEngine):
    """
    Maximum a Posteriori (MAP) Inference Engine.

    This class performs MAP inference using Stochastic Variational Inference (SVI)
    with AutoDelta guide. It provides methods for inference and prediction.

    Parameters
    ----------
    model : Callable
        The probabilistic model to perform inference on.
    optimizer_factory : numpyro.optim._NumPyroOptim, optional
        The optimizer to use for SVI. Defaults to None.
    num_steps : int, optional
        The number of optimization steps to perform. Defaults to 10000.
    rng_key : jax.random.PRNGKey, optional
        The random number generator key. Defaults to None.
    """

    _tags = {
        "inference_method": "map",
    }

    def __init__(
        self,
        optimizer_factory: numpyro.optim._NumPyroOptim = None,
        optimizer: Optional[BaseOptimizer] = None,
        num_steps=10_000,
        num_samples=_DEFAULT_PREDICT_NUM_SAMPLES,
        rng_key=None,
        progress_bar: bool = DEFAULT_PROGRESS_BAR,
        stable_update=False,
    ):

        self.optimizer_factory = optimizer_factory
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.progress_bar = progress_bar
        self.stable_update = stable_update
        super().__init__(rng_key)

        deprecation_warning(
            "optimizer_factory",
            "0.5.0",
            "Please use the `optimizer` parameter instead.",
        )

        if optimizer_factory is None and optimizer is None:
            optimizer = AdamOptimizer(1e-3)

        if self.optimizer is None and optimizer_factory is not None:
            optimizer = _OptimizerFromCallable(optimizer_factory)

        self._optimizer = optimizer

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
        self.guide_ = AutoDelta(self.model_, init_loc_fn=init_to_mean())

        def get_result(
            rng_key,
            model,
            guide,
            optimizer,
            num_steps,
            progress_bar,
            stable_update,
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
                **kwargs,
            )

        self.run_results_: SVIRunResult = get_result(
            self._rng_key,
            self.model_,
            self.guide_,
            self._optimizer.create_optimizer(),
            self.num_steps,
            stable_update=self.stable_update,
            progress_bar=self.progress_bar,
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
        numpyro.samples_
            The predicted samples generated by the model.
        """
        predictive = numpyro.infer.Predictive(
            self.model_,
            params=self.run_results_.params,
            guide=self.guide_,
            # posterior_samples=self.posterior_samples_,
            num_samples=self.num_samples,
        )
        numpyro.samples_ = predictive(rng_key=self._rng_key, **kwargs)
        return numpyro.samples_


class MAPInferenceEngineError(Exception):
    """Exception raised for NaN losses in MAPInferenceEngine."""

    def __init__(self, message="NaN losses in MAPInferenceEngine"):
        self.message = message
        super().__init__(self.message)
