"""Extension template for creating a new effect in Prophetverse."""

from typing import Any, Dict, Optional
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from prophetverse.effects.base import BaseEffect
from prophetverse.utils.frame_to_array import series_to_tensor_or_array


class MySimpleEffectName(BaseEffect):
    """
    A simple custom effect example that only overrides `_predict`.

    This template is suitable when no fitting or parameter sampling is required,
    and the effect is a direct transformation of the input data.

    Parameters
    ----------
    param1 : float
        A scaling factor applied to the input data.
    param2 : float
        A constant bias added to the scaled input.

    `_transform` in BaseEffect may return one of:
      - `jnp.ndarray`
      - `tuple` with first element a `jnp.ndarray` and extra metadata
      - `dict` containing key "data" with a `jnp.ndarray` value
      - `list` of any of the above, for broadcasted inputs

    Tag behavior:
      - `capability:panel`:
          If True, supports panel (multiple time series) input in one call. False here.
      - `capability:multivariate_input`:
          If True, `_transform` can ingest multiple columns at once. False here triggers broadcasting.
      - `requires_X`:
          If True, skips effect if no X columns found. False would allow effect without X.
      - `applies_to`:
          Indicates whether this effect applies to 'X' (exogenous) or 'y' (target).
      - `filter_indexes_with_forecating_horizon_at_transform`:
          If True, filters X to only the forecasting horizon before `_transform`.
      - `requires_fit_before_transform`:
          If True, raises if `transform` is called before `fit`.
    """

    _tags = {
        "capability:panel": False,  # no panel/multi-series support
        "capability:multivariate_input": False,  # single-column input only
        "requires_X": True,  # needs exogenous X
        "applies_to": "X",  # effect applies to X
        "filter_indexes_with_forecating_horizon_at_transform": True,  # filter to fh
        "requires_fit_before_transform": False,  # transform does not require prior fit
    }

    def __init__(self, param1: float = 1.0, param2: float = 0.0):
        # assign hyperparameters before BaseEffect init
        self.param1 = param1
        self.param2 = param2
        super().__init__()

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Compute the custom effect by scaling `data` and adding a bias.

        Parameters
        ----------
        data : jnp.ndarray
            Transformed exogenous data, shape (T, 1) or (N, T, 1).
        predicted_effects : dict
            Other effect arrays (unused in this simple example).

        Returns
        -------
        jnp.ndarray
            The effect contribution, same shape as `data`.
        """
        # TODO: Replace this with your own transformation logic if needed
        return data * self.param1 + self.param2


class MyEffectName(BaseEffect):
    """
    A full-featured custom effect example.

    Steps to implement a new effect:
      1. Override `_fit` (optional) to compute static quantities from `y` and `X`.
      2. Override `_transform` (optional) to prepare `X` as JAX arrays.
      3. Within `_predict`, sample any parameters via `numpyro.sample`
      4. Implement `_predict` (required) using `data`, `predicted_effects`, and samples.

    Parameters
    ----------
    param1 : float
        A multiplier applied in `_predict`.
    param2 : float
        A bias term added in `_predict`.
    prior_scale : float
        Scale of the Normal prior for sampling a coefficient.

    Notes
    -----
    `_transform` may return:
      - `jnp.ndarray`
      - `tuple` for arrays plus metadata
      - `dict` with key "data"
      - `list` of the above (for broadcasted columns)

    Tag behavior:
      - `capability:panel`: supports panel input if True
      - `capability:multivariate_input`: allow multi-column transform if True
      - `requires_X`: skip effect if X not present when True
      - `applies_to`: "X" or "y" determines which DataFrame the transform applies to
      - `filter_indexes_with_forecating_horizon_at_transform`: filter to fh before `_transform`
      - `requires_fit_before_transform`: if True, `transform` errors when called pre-`fit`
    """

    _tags = {
        "capability:panel": False,  # no built-in panel support
        "capability:multivariate_input": False,  # will broadcast columns by default
        "requires_X": True,  # needs X; if missing, skips predict
        "applies_to": "X",  # transform applies to exogenous inputs
        "filter_indexes_with_forecating_horizon_at_transform": True,
        "requires_fit_before_transform": False,
    }

    def __init__(
        self,
        param1: float = 1.0,
        param2: float = 0.0,
        prior_scale: float = 1.0,
    ):
        # set hyperparameters and priors before base init
        self.param1 = param1
        self.param2 = param2
        self._param_prior = dist.Normal(0.0, prior_scale)
        super().__init__()

    def _fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame], scale: float = 1.0):
        """
        Optional: called during fit to compute static quantities.

        Example implementation centers X by column means.
        """
        if X is not None:
            # compute and store column means of X for centering
            self._X_mean = X.mean()
            # TODO: add additional fit logic here (e.g., compute other summaries)
        else:
            self._X_mean = None

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Any:
        """
        Optional: prepare X for the model.

        Example implementation subtracts stored means then converts to tensor.
        """
        if self._X_mean is not None:
            # center X
            X_proc = X - self._X_mean
            # TODO: add any other transform steps (e.g., feature engineering)
        else:
            X_proc = X
        # convert to JAX tensor/array
        return series_to_tensor_or_array(X_proc)

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Core effect logic: sample parameters and compute contribution.
        """
        # sample a coefficient from the prior
        coef = numpyro.sample("coef", self._param_prior)
        # TODO: sample additional parameters if your effect needs more

        # unwrap array from dict/tuple if necessary
        if isinstance(data, dict):
            arr = data["data"]
        elif isinstance(data, tuple):
            arr = data[0]
        else:
            arr = data

        # compute the effect: data * coef * param1 + param2
        # TODO: replace with your own custom computation
        return arr * coef * self.param1 + self.param2

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """
        Return test parameters for automated framework checks.
        """
        return [{"param1": 2.0, "param2": 0.5, "prior_scale": 1.0}]
