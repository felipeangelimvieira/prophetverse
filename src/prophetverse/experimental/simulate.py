"""Simulate data from a model, and intervene optionally."""

from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax.random import PRNGKey

from prophetverse.sktime.base import BaseProphetForecaster


def simulate(
    model: BaseProphetForecaster,
    fh: pd.Index,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.DataFrame] = None,
    do: Optional[Dict[str, Union[jnp.ndarray, float]]] = None,
    num_samples: int = 10,
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

    Returns
    -------
    Dict
        A dictionary with the simulated data.
    """
    # Fit model, creating a dummy y if it is not provided
    if y is None:
        y = pd.DataFrame(index=fh, data=np.random.rand(len(fh)) * 10, columns=["dummy"])

    model.fit(X=X, y=y)

    # Get predict data to call predictive model
    predict_data = model._get_predict_data(X=X, fh=fh)
    predict_data["y"] = None
    from numpyro.infer import Predictive

    predictive_model = model.model
    if do is not None:
        predictive_model = numpyro.handlers.do(predictive_model, data=do)

    # predictive_model = model.model
    predictive_model = Predictive(model=predictive_model, num_samples=num_samples)
    predictive_output = predictive_model(PRNGKey(0), **predict_data)
    return predictive_output
