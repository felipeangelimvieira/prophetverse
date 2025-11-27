"""Numpyro VI inference engines for prophet models.

The VIInferenceEngine class performs Variational Inference using SVI with
configurable autoguides specified as string parameters.
"""

from typing import Optional, Union, Callable

import jax.numpy as jnp
import jax.random
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoGuide,
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.infer.initialization import init_to_mean
from numpyro.infer.svi import SVIRunResult

from prophetverse.engine.base import BaseInferenceEngine
from prophetverse.engine.optimizer.optimizer import (
    CosineScheduleAdamOptimizer,
    BaseOptimizer,
)

_DEFAULT_PREDICT_NUM_SAMPLES = 1000
DEFAULT_PROGRESS_BAR = False

# Mapping of guide string names to numpyro autoguide classes
GUIDE_MAP = {
    "AutoNormal": AutoNormal,
    "AutoMultivariateNormal": AutoMultivariateNormal,
    "AutoDiagonalNormal": AutoDiagonalNormal,
    "AutoLowRankMultivariateNormal": AutoLowRankMultivariateNormal,
}


class VIInferenceEngine(BaseInferenceEngine):
    """
    Variational Inference Engine.

    This class performs Variational Inference using Stochastic Variational Inference (SVI)
    with configurable autoguides. It provides methods for inference and prediction.

    Parameters
    ----------
    guide : str or AutoGuide, optional
        The name of the autoguide to use for variational inference.
        Available options: "AutoNormal", "AutoMultivariateNormal",
        "AutoDiagonalNormal", "AutoLowRankMultivariateNormal".
        Default is "AutoNormal".
    optimizer : Optional[BaseOptimizer]
        The optimizer to be used for inference. If not provided, the default is
        AdamOptimizer.
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
    init_scale : float, optional
        The scale for initializing the parameters. Default is 0.1.
    init_loc_fn : optional
        The function to initialize the location parameter. If not provided, the default is init_to_mean.

    """

    _tags = {
        "inference_method": "vi",
    }

    def __init__(
        self,
        guide: Optional[Union[str, Callable]] = "AutoDiagonalNormal",
        optimizer: Optional[BaseOptimizer] = None,
        num_steps=10_000,
        num_samples=_DEFAULT_PREDICT_NUM_SAMPLES,
        rng_key=None,
        progress_bar: bool = DEFAULT_PROGRESS_BAR,
        stable_update=False,
        forward_mode_differentiation=False,
        init_scale=0.1,
        init_loc_fn=None,
    ):
        self.guide = guide
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.progress_bar = progress_bar
        self.stable_update = stable_update
        self.forward_mode_differentiation = forward_mode_differentiation
        self.init_scale = init_scale
        self.init_loc_fn = init_loc_fn
        super().__init__(rng_key)

        # Validate guide parameter
        if guide not in GUIDE_MAP:
            available_guides = list(GUIDE_MAP.keys())
            raise ValueError(
                f"Unknown guide '{guide}'. Available guides: {available_guides}"
            )

        if optimizer is None:
            optimizer = CosineScheduleAdamOptimizer()

        self._optimizer = optimizer
        self._guide_class = GUIDE_MAP[guide] if isinstance(guide, str) else None

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
        Perform Variational Inference.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        self
            The updated VIInferenceEngine object.
        """
        self.guide_ = self._guide_class(
            self.model_, init_loc_fn=self._init_loc_fn, init_scale=self.init_scale
        )

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

        # Split RNG key for inference to ensure reproducibility
        self._rng_key, infer_key = jax.random.split(self._rng_key)
        
        self.run_results_: SVIRunResult = get_result(
            infer_key,
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

        # Split RNG key for sampling posterior to ensure reproducibility
        self._rng_key, sample_key = jax.random.split(self._rng_key)
        
        self.posterior_samples_ = self.guide_.sample_posterior(
            sample_key, params=self.run_results_.params, **kwargs
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
        VIInferenceEngineError
            If the last loss is NaN.
        """
        losses = run_results.losses
        if jnp.isnan(losses)[-1]:
            msg = "NaN losses in VIInferenceEngine."
            msg += " Try decreasing the learning rate or changing the model specs."
            msg += " If the problem persists, please open an issue at"
            msg += " https://github.com/felipeangelimvieira/prophetverse"
            raise VIInferenceEngineError(msg)

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
        # Split RNG key for prediction to ensure reproducibility
        self._rng_key, predict_key = jax.random.split(self._rng_key)
        
        predictive = numpyro.infer.Predictive(
            self.model_,
            params=self.run_results_.params,
            guide=self.guide_,
            num_samples=self.num_samples,
        )
        self.samples_ = predictive(rng_key=predict_key, **kwargs)
        return self.samples_

    @classmethod
    def get_test_params(*args, **kwargs):
        """Return test params for unit testing."""
        return [
            {
                "guide": "AutoNormal",
                "optimizer": CosineScheduleAdamOptimizer(),
                "num_steps": 100,
            },
            {
                "guide": "AutoMultivariateNormal",
                "optimizer": CosineScheduleAdamOptimizer(),
                "num_steps": 100,
            },
            {
                "guide": "AutoDiagonalNormal",
                "optimizer": CosineScheduleAdamOptimizer(),
                "num_steps": 100,
            },
        ]


class VIInferenceEngineError(Exception):
    """Exception raised for NaN losses in VIInferenceEngine."""

    def __init__(self, message="NaN losses in VIInferenceEngine"):
        self.message = message
        super().__init__(self.message)
