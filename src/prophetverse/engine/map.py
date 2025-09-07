"""Numpyro inference engines for prophet models.

The classes in this module take a model, the data and perform inference using Numpyro.
"""

from typing import Optional

from prophetverse.engine.vi import VIInferenceEngine, VIInferenceEngineError
from prophetverse.engine.optimizer.optimizer import (
    AdamOptimizer,
    BaseOptimizer,
    LBFGSSolver,
)

_DEFAULT_PREDICT_NUM_SAMPLES = 1000
DEFAULT_PROGRESS_BAR = False


class MAPInferenceEngineError(VIInferenceEngineError):
    """Exception raised for NaN losses in MAPInferenceEngine."""

    def __init__(self, message="NaN losses in MAPInferenceEngine"):
        super().__init__(message)


class MAPInferenceEngine(VIInferenceEngine):
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
    _exc_class = MAPInferenceEngineError

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
        if optimizer is None:
            optimizer = LBFGSSolver()

        super().__init__(
            guide="AutoDelta",
            optimizer=optimizer,
            num_steps=num_steps,
            num_samples=num_samples,
            rng_key=rng_key,
            progress_bar=progress_bar,
            stable_update=stable_update,
            forward_mode_differentiation=forward_mode_differentiation,
            init_loc_fn=init_loc_fn,
        )

        if self._optimizer.get_tag("is_solver", False):  # type: ignore[union-attr]
            # If solver, there's a single "solver step". For compatibility,
            # we set num_steps to 1 and max_iter to the original num_steps.

            self._optimizer = self._optimizer.set_max_iter(  # type: ignore[union-attr]
                self._num_steps
            )
            self._num_steps = 1

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
