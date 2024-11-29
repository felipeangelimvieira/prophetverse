# Customizing exogenous effects

The exogenous effect API allows you to create custom exogenous components for the Prophetverse model. This is useful when we want to model specific patterns or relationships between the exogenous variables and the target variable. For example, enforcing a positive effect of a variable on the mean, or modeling a non-linear relationship.

If you have read the [theory section](https://prophetverse.com/the-theory/), 
by effect we mean each function $f_i$. You can implement those custom
functions by subclassing the `BaseEffect` class, and then use them in the
`Prophetverse` model. Some effects are already implemented in the library, 
and you can find them in the `prophetverse.effects` module.

When creating a model instance, effects can be specified through `exogenous_effects` 
parameter of the `Prophetverse` model. This parameter is a list of tuples of three 
values: the name, the effect object, and a regex to filter
columns related to that effect. The regex is what defines $x_i$ in the previous section. The `prophetverse.utils.regex` module provides some useful functions to create regex patterns for common use cases, include `starts_with`, `ends_with`, `contains`, and `no_input_columns`.

For example:




```python
from prophetverse.sktime import Prophetverse
from prophetverse.effects import LinearFourierSeasonality, HillEffect
from prophetverse.utils.regex import starts_with, no_input_columns, exact


exogenous_effects = [
    (
        "seasonality",  # The name of the effect
        LinearFourierSeasonality(  # The object
            freq="D",
            sp_list=[7, 365.25],
            fourier_terms_list=[3, 10],
            prior_scale=0.1,
            effect_mode="multiplicative",
        ),
        no_input_columns,  # The regex
    ),
    (
        "channel1_investment_incremental", # The name of the effect
        HillEffect(
            effect_mode="additive"
            ),
        exact("channel1_investment"), # Column in dataframe
    ),
]

model = Prophetverse(exogenous_effects=exogenous_effects)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[1]</span></p>


The effects can be any object that implements the `BaseEffect` interface, and you can
create your own effects by subclassing `BaseEffect` and implementing `_fit`, `_transform` and
`_predict` methods.

* `_fit` (optional): This method is called during fit() of the forecasting  and should
be used to initialize any necessary parameters or data structures.
It receives the exogenous variables dataframe X, the series `y`, and the scale factor
`scale` that was used to scale the timeseries.

* `_transform` (optional): This method receives the exogenous variables dataframe,
and should return an object containing the data needed for the effect. This object 
will be passed to the predict method as `data`. By default the columns of the 
dataframe that match the regex pattern are selected, and the result is converted to
a `jnp.ndarray`.

* `_predict` (mandatory): This method receives the output of `_transform` and all 
previously computed effects. It should return the effect values as a `jnp.ndarray`

In many cases, the `_fit` and `_transform` steps are not needed to be implemented,
since the default behaviour may be the desired one. In the example below, we implement
a really simple `LogEffect` class, which leverages the default behaviour of the 
`BaseEffect` class.

## Example

### Log Effect

The `BaseAdditiveOrMultiplicativeEffect` provides an init argument `effect_mode` that
allows you to specify if the effect is additive or multiplicative. Let's take as an 
example the `LogEffect`:







```python

# prophetverse/effects/log.py

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution

from prophetverse.effects.base import (
    BaseEffect,
)

class LogEffect(BaseEffect):
    """Represents a log effect as effect = scale * log(rate * data + 1).

    Parameters
    ----------
    scale_prior : Optional[Distribution], optional
        The prior distribution for the scale parameter., by default Gamma
    rate_prior : Optional[Distribution], optional
        The prior distribution for the rate parameter., by default Gamma
    effect_mode : effects_application, optional
        Either "additive" or "multiplicative", by default "multiplicative"
    """

    def __init__(
        self,
        scale_prior: Optional[Distribution] = None,
        rate_prior: Optional[Distribution] = None,
    ):
        self.scale_prior = scale_prior or dist.Gamma(1, 1)
        self.rate_prior = rate_prior or dist.Gamma(1, 1)
        super().__init__()

    def _predict(  # type: ignore[override]
        self,
        data: jnp.ndarray,
        predicted_effects: Optional[Dict[str, jnp.ndarray]] = None,
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
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        scale = numpyro.sample("log_scale", self.scale_prior)
        rate = numpyro.sample("log_rate", self.rate_prior)
        effect = scale * jnp.log(jnp.clip(rate * data + 1, 1e-8, None))

        return effect


```



The `_fit` and `_transform` methods are not implemented, and the default behaviour is
preserved (the columns of the dataframe that match the regex pattern are selected, and the result is converted to a `jnp.ndarray` with key "data").



