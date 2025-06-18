"""Simulate data from a model, and intervene optionally."""

from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax.random import PRNGKey

from prophetverse.sktime.base import BaseProphetForecaster
from prophetverse._model import model as model_func
from prophetverse.effects.target.base import BaseTargetEffect

from numpyro.primitives import Messenger


class IgnoreObservedSites(Messenger):
    def __init__(self):
        """
        obs_map: a dict mapping site names -> values
        """
        super().__init__(fn=None)

    def process_message(self, msg):
        # only intercept real sample sites that have no value yet

        if msg["type"] == "sample":
            # ...but tell NumPyro it's NOT an observed site
            msg["is_observed"] = False
            msg["obs"] = None


def simulate(
    model: BaseProphetForecaster,
    fh: pd.Index,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.DataFrame] = None,
    do: Optional[Dict[str, Union[jnp.ndarray, float]]] = None,
    return_model=False,
):
    """
    Simulate data from a model.

    **EXPERIMENTAL FEATURE**
    This feature allow to do prior predictive checks and to intervene to
    obtain simulated data.

    Parameters
    ----------
    model : BaseProphetForecaster
        The probabilistic model to perform inference on.
    fh : pd.Index
        The forecasting horizon as a pandas Index.
    X : pd.DataFrame, optional
        The input DataFrame containing the exogenous variables.
    y : pd.DataFrame, optional
        The timeseries dataframe. This is used by effects that implement `_fit` and
        use the target timeseries to initialize some parameters. If not provided,
        a dummy y will be created.
    do : Dict, optional
        A dictionary with the variables to intervene and their values.
    num_samples : int, optional
        The number of samples to generate. Defaults to 10.
    return_model : bool, optional
        If True, the fitted model will be returned. Defaults to False.
    Returns
    -------
    Union[Dict, Tuple]
        If return_model=False, a dictionary with the simulated data. Otherwise,
        a tuple (simulated_data, model).
    """
    # Fit model, creating a dummy y if it is not provided
    if y is None:
        if X is None:
            index = fh
        else:
            index, _ = X.index.reindex(fh, level=-1)
        y = pd.DataFrame(
            index=index, data=np.random.rand(len(index)) * 10, columns=["dummy"]
        )

    model = model.clone()
    model.fit(X=X, y=y)

    with IgnoreObservedSites():
        if do is not None:
            with numpyro.handlers.do(data=do):
                components = model.predict_component_samples(X=X, fh=fh)
        else:
            components = model.predict_component_samples(X=X, fh=fh)
    if return_model:
        return components, model
    return components
