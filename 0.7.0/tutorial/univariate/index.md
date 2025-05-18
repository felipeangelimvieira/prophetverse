# Basic univariate forecasting
This example shows how to use Prophetverse to perform univariate forecasting
with a time series dataset, using sktime-style interface

Because of this compatibility, you can benefit from all the features of sktime, such
as hierarchical reconciliation, ensemble models, pipelines, etc. There are two main
methods to use Prophetverse with sktime:

* `fit(y, X=None)`:  This method is used to fit the model. It takes as input
a time series `y` and an optional exogenous variable `X`. The `y` time series must be
a `pd.Series` or a `pd.DataFrame`. The `X` variable must be a `pd.DataFrame` or `None`.

* `predict(fh, X=None)`: This method is used to make predictions. It takes as input a
forecast horizon `fh` and an optional exogenous variable `X`. The `fh` forecast horizon
can be a relative or an absolute forecast horizon. The `X` variable must be a
`pd.DataFrame` or `None`, according to the `X` variable used in the `fit` method.

Later in this example, we will also show additional methods to make predictions, such as
`predict_quantiles` and `predict_components`.



```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpyro import distributions as dist



```

## Import dataset

We import a dataset from Prophet's original repository. We then put it into sktime-friendly format, where the index is a `pd.PeriodIndex` and the colums are the time series.




```python

from prophetverse.datasets.loaders import load_peyton_manning

y = load_peyton_manning()
display(y.head())


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[3]</span></p>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007-12-10</th>
      <td>9.590761</td>
    </tr>
    <tr>
      <th>2007-12-11</th>
      <td>8.519590</td>
    </tr>
    <tr>
      <th>2007-12-12</th>
      <td>8.183677</td>
    </tr>
    <tr>
      <th>2007-12-13</th>
      <td>8.072467</td>
    </tr>
    <tr>
      <th>2007-12-14</th>
      <td>7.893572</td>
    </tr>
  </tbody>
</table>
</div>


The full dataset looks like this:



```python
y.plot.line(figsize=(12, 6))
plt.show()



```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[4]</span></p>


    
![png](index_files/output_5_0.png)
    


## Fit model
Here, we will show how you can fit a simple model with Prophetverse.
We first fit a model without seasonal components, and then fit a full model.
We also show how easy it is to switch between Maximum A Posteriori (MAP) inference
and Markov Chain Monte Carlo (MCMC).

### No seasonality




```python
from prophetverse.effects.trend import PiecewiseLinearTrend
from prophetverse.engine import MAPInferenceEngine
from prophetverse.sktime import Prophetverse
from prophetverse.utils import no_input_columns

model = Prophetverse(
    trend=PiecewiseLinearTrend(
        changepoint_interval=500,
        changepoint_prior_scale=0.00001,
        changepoint_range=-250,
    ),
    inference_engine=MAPInferenceEngine(),
)
model.fit(y=y)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[5]</span></p>




<style>#sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c {
    /* Definition of color scheme common for light and dark mode */
    --sklearn-color-text: black;
    --sklearn-color-line: gray;
    /* Definition of color scheme for objects */
    --sklearn-color-level-0: #fff5e6;
    --sklearn-color-level-1: #f6e4d2;
    --sklearn-color-level-2: #ffe0b3;
    --sklearn-color-level-3: chocolate;

    /* Specific color for light theme */
    --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, white));
    --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-icon: #696969;

    @media (prefers-color-scheme: dark) {
      /* Redefinition of color scheme for dark theme */
      --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, #111));
      --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-icon: #878787;
    }
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c {
    color: var(--sklearn-color-text);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c pre {
    padding: 0;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c input.sk-hidden--visually {
    border: 0;
    clip: rect(1px 1px 1px 1px);
    clip: rect(1px, 1px, 1px, 1px);
    height: 1px;
    margin: -1px;
    overflow: hidden;
    padding: 0;
    position: absolute;
    width: 1px;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-text-repr-fallback {
    display: none;
  }

  div.sk-parallel-item,
  div.sk-serial,
  div.sk-item {
    /* draw centered vertical line to link estimators */
    background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
    background-size: 2px 100%;
    background-repeat: no-repeat;
    background-position: center center;
  }

  /* Parallel-specific style estimator block */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-serial {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--sklearn-color-background);
    padding-right: 1em;
    padding-left: 1em;
  }


  /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
  clickable and can be expanded/collapsed.
  - Pipeline and ColumnTransformer use this feature and define the default style
  - Estimators will overwrite some part of the style using the `sk-estimator` class
  */

  /* Pipeline and ColumnTransformer style (default) */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label label.sk-toggleable__label,
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c div.sk-estimator:hover {
    background-color: var(--sklearn-color-level-2);
  }

  /* Specification for estimator info */

  .sk-estimator-doc-link,
  a:link.sk-estimator-doc-link,
  a:visited.sk-estimator-doc-link {
    float: right;
    font-size: smaller;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1em;
    height: 1em;
    width: 1em;
    text-decoration: none !important;
    margin-left: 1ex;
    border: var(--sklearn-color-level-1) 1pt solid;
    color: var(--sklearn-color-level-1);
  }

  /* On hover */
  div.sk-estimator:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover,
  div.sk-label-container:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }

  /* Span, style for the box shown on hovering the info icon */
  .sk-estimator-doc-link span {
    display: none;
    z-index: 9999;
    position: relative;
    font-weight: normal;
    right: .2ex;
    padding: .5ex;
    margin: .5ex;
    width: min-content;
    min-width: 20ex;
    max-width: 50ex;
    color: var(--sklearn-color-text);
    box-shadow: 2pt 2pt 4pt #999;
    background: var(--sklearn-color-level-0);
    border: .5pt solid var(--sklearn-color-level-3);
  }

  .sk-estimator-doc-link:hover span {
    display: block;
  }

  /* "?"-specific style due to the `<a>` HTML tag */

  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c a.estimator_doc_link {
    float: right;
    font-size: 1rem;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1rem;
    height: 1rem;
    width: 1rem;
    text-decoration: none;
    color: var(--sklearn-color-level-1);
    border: var(--sklearn-color-level-1) 1pt solid;
  }

  /* On hover */
  #sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-ac145de4-7e42-4c72-9b00-c64a5bb1a56c' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Prophetverse(inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-250))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('aa0015d6-c62c-4af4-a6fc-fd629d4a529d') type="checkbox" ><label for=UUID('aa0015d6-c62c-4af4-a6fc-fd629d4a529d') class='sk-toggleable__label sk-toggleable__label-arrow'>Prophetverse</label><div class="sk-toggleable__content"><pre>Prophetverse(inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-250))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>effects</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('9458bd91-887a-4473-95e2-31c95d28abcb') type="checkbox" ><label for=UUID('9458bd91-887a-4473-95e2-31c95d28abcb') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLinearTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLinearTrend(changepoint_interval=500, changepoint_prior_scale=1e-05,
                     changepoint_range=-250)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>inference_engine</label></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('4be83960-907e-40dc-be3c-0477e587d04e') type="checkbox" ><label for=UUID('4be83960-907e-40dc-be3c-0477e587d04e') class='sk-toggleable__label sk-toggleable__label-arrow'>MAPInferenceEngine</label><div class="sk-toggleable__content"><pre>MAPInferenceEngine()</pre></div></div></div></div></div></div></div></div></div></div>




```python
forecast_horizon = pd.period_range("2007-01-01", "2018-01-01", freq="D")
fig, ax = plt.subplots(figsize=(10, 5))
preds = model.predict(fh=forecast_horizon)
preds.plot.line(ax=ax)
ax.scatter(y.index, y, marker="o", color="k", s=2, alpha=0.5)



```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[6]</span></p>




    <matplotlib.collections.PathCollection at 0x319c97a10>




    
![png](index_files/output_8_1.png)
    


### With seasonality
Here, we fit the univariate Prophet and pass an exogenous effect as hyperparameter.
The `exogenous_effects` parameter let us add new components to the model and control
the relationship between exogenous variables and the target variable.
In this case, the `LinearFourierSeasonality` effect creates sinusoidal and cosine
terms to model the seasonality of the time series, which are then multiplied by
linear coefficients and added to the model.

This argument is a list of tuples of the form `(effect_name, effect, regex_to_filter_relevant_columns)`,
where `effect_name` is a string and `effect` is an instance of a subclass of
`prophetverse.effects.BaseEffect`. The regex is used to filter the columns of X
that are relevant for the effect, but can also be `None`
(or its alias `prophetverse.utils.no_input_columns`) if no input in `X` is needed
for the effect. For example, the seasonality effect already implemented in
`prophetverse.effects` module does not need any input in `X`, so we can use
`prophetverse.utils.no_input_columns` as the regex.




```python
from prophetverse.effects.fourier import LinearFourierSeasonality
from prophetverse.utils import no_input_columns

model = Prophetverse(
    trend=PiecewiseLinearTrend(
        changepoint_interval=500,
        changepoint_prior_scale=0.00001,
        changepoint_range=-500,
    ),
    exogenous_effects=[
        (
            "seasonality",
            LinearFourierSeasonality(
                freq="D",
                sp_list=[7, 365.25],
                fourier_terms_list=[3, 10],
                prior_scale=0.1,
                effect_mode="multiplicative",
            ),
            no_input_columns,
        ),
    ],
    inference_engine=MAPInferenceEngine(),
)
model.fit(y=y)



```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[7]</span></p>




<style>#sk-6c62a257-56ca-4d35-888e-985d0a5c4315 {
    /* Definition of color scheme common for light and dark mode */
    --sklearn-color-text: black;
    --sklearn-color-line: gray;
    /* Definition of color scheme for objects */
    --sklearn-color-level-0: #fff5e6;
    --sklearn-color-level-1: #f6e4d2;
    --sklearn-color-level-2: #ffe0b3;
    --sklearn-color-level-3: chocolate;

    /* Specific color for light theme */
    --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, white));
    --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-icon: #696969;

    @media (prefers-color-scheme: dark) {
      /* Redefinition of color scheme for dark theme */
      --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, #111));
      --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-icon: #878787;
    }
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 {
    color: var(--sklearn-color-text);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 pre {
    padding: 0;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 input.sk-hidden--visually {
    border: 0;
    clip: rect(1px 1px 1px 1px);
    clip: rect(1px, 1px, 1px, 1px);
    height: 1px;
    margin: -1px;
    overflow: hidden;
    padding: 0;
    position: absolute;
    width: 1px;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-text-repr-fallback {
    display: none;
  }

  div.sk-parallel-item,
  div.sk-serial,
  div.sk-item {
    /* draw centered vertical line to link estimators */
    background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
    background-size: 2px 100%;
    background-repeat: no-repeat;
    background-position: center center;
  }

  /* Parallel-specific style estimator block */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-serial {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--sklearn-color-background);
    padding-right: 1em;
    padding-left: 1em;
  }


  /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
  clickable and can be expanded/collapsed.
  - Pipeline and ColumnTransformer use this feature and define the default style
  - Estimators will overwrite some part of the style using the `sk-estimator` class
  */

  /* Pipeline and ColumnTransformer style (default) */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label label.sk-toggleable__label,
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 div.sk-estimator:hover {
    background-color: var(--sklearn-color-level-2);
  }

  /* Specification for estimator info */

  .sk-estimator-doc-link,
  a:link.sk-estimator-doc-link,
  a:visited.sk-estimator-doc-link {
    float: right;
    font-size: smaller;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1em;
    height: 1em;
    width: 1em;
    text-decoration: none !important;
    margin-left: 1ex;
    border: var(--sklearn-color-level-1) 1pt solid;
    color: var(--sklearn-color-level-1);
  }

  /* On hover */
  div.sk-estimator:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover,
  div.sk-label-container:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }

  /* Span, style for the box shown on hovering the info icon */
  .sk-estimator-doc-link span {
    display: none;
    z-index: 9999;
    position: relative;
    font-weight: normal;
    right: .2ex;
    padding: .5ex;
    margin: .5ex;
    width: min-content;
    min-width: 20ex;
    max-width: 50ex;
    color: var(--sklearn-color-text);
    box-shadow: 2pt 2pt 4pt #999;
    background: var(--sklearn-color-level-0);
    border: .5pt solid var(--sklearn-color-level-3);
  }

  .sk-estimator-doc-link:hover span {
    display: block;
  }

  /* "?"-specific style due to the `<a>` HTML tag */

  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 a.estimator_doc_link {
    float: right;
    font-size: 1rem;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1rem;
    height: 1rem;
    width: 1rem;
    text-decoration: none;
    color: var(--sklearn-color-level-1);
    border: var(--sklearn-color-level-1) 1pt solid;
  }

  /* On hover */
  #sk-6c62a257-56ca-4d35-888e-985d0a5c4315 a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-6c62a257-56ca-4d35-888e-985d0a5c4315' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-500))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('c2fc5c61-9953-4a9b-b7e7-87812265348c') type="checkbox" ><label for=UUID('c2fc5c61-9953-4a9b-b7e7-87812265348c') class='sk-toggleable__label sk-toggleable__label-arrow'>Prophetverse</label><div class="sk-toggleable__content"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-500))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>effects</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('1af11104-f9dd-4b35-89d1-db8af930d55a') type="checkbox" ><label for=UUID('1af11104-f9dd-4b35-89d1-db8af930d55a') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLinearTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLinearTrend(changepoint_interval=500, changepoint_prior_scale=1e-05,
                     changepoint_range=-500)</pre></div></div></div><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('f2e6cb30-5c6c-478d-95c0-2b34120f72fb') type="checkbox" ><label for=UUID('f2e6cb30-5c6c-478d-95c0-2b34120f72fb') class='sk-toggleable__label sk-toggleable__label-arrow'>LinearFourierSeasonality</label><div class="sk-toggleable__content"><pre>LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                         fourier_terms_list=[3, 10], freq=&#x27;D&#x27;, prior_scale=0.1,
                         sp_list=[7, 365.25])</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>inference_engine</label></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('ee554e03-2c47-4eda-a7f4-af99ae525612') type="checkbox" ><label for=UUID('ee554e03-2c47-4eda-a7f4-af99ae525612') class='sk-toggleable__label sk-toggleable__label-arrow'>MAPInferenceEngine</label><div class="sk-toggleable__content"><pre>MAPInferenceEngine()</pre></div></div></div></div></div></div></div></div></div></div>




```python
forecast_horizon = pd.period_range("2007-01-01", "2018-01-01", freq="D")
fig, ax = plt.subplots(figsize=(10, 5))
preds = model.predict(fh=forecast_horizon)
preds.plot.line(ax=ax)
ax.scatter(y.index, y, marker="o", color="k", s=2, alpha=0.5)



```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[8]</span></p>




    <matplotlib.collections.PathCollection at 0x322e2e950>




    
![png](index_files/output_11_1.png)
    


## Probabilistic forecasting
We can also make probabilistic forecasts with Prophetverse, in sktime fashion.
 The `predict_quantiles` method returns the quantiles of the predictive distribution
in a `pd.DataFrame`



```python
quantiles = model.predict_quantiles(fh=forecast_horizon, alpha=[0.1, 0.9])
quantiles.head()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[9]</span></p>




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">y</th>
    </tr>
    <tr>
      <th></th>
      <th>0.1</th>
      <th>0.9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007-01-01</th>
      <td>8.066615</td>
      <td>9.367413</td>
    </tr>
    <tr>
      <th>2007-01-02</th>
      <td>7.873573</td>
      <td>9.173687</td>
    </tr>
    <tr>
      <th>2007-01-03</th>
      <td>7.761128</td>
      <td>9.031336</td>
    </tr>
    <tr>
      <th>2007-01-04</th>
      <td>7.715733</td>
      <td>9.067670</td>
    </tr>
    <tr>
      <th>2007-01-05</th>
      <td>7.747439</td>
      <td>9.096114</td>
    </tr>
  </tbody>
</table>
</div>



The plot below shows the (0.1, 0.9) quantiles of the predictive distribution



```python
fig, ax = plt.subplots(figsize=(10, 5))
# Plot area between quantiles
ax.fill_between(
    quantiles.index.to_timestamp(),
    quantiles.iloc[:, 0],
    quantiles.iloc[:, -1],
    alpha=0.5,
)
ax.scatter(y.index, y, marker="o", color="k", s=2, alpha=1)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[10]</span></p>




    <matplotlib.collections.PathCollection at 0x324367b90>




    
![png](index_files/output_15_1.png)
    


## Timeseries decomposition
We can easily extract the components of the time series with the `predict_components`
method. This method, in particular, is not implemented in sktime's `BaseForecaster`,
but it is a method of `prophetverse.Prophetverse` class.



```python
sites = model.predict_components(fh=forecast_horizon)
sites.head()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[11]</span></p>




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>obs</th>
      <th>seasonality</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2007-01-01</th>
      <td>8.716851</td>
      <td>8.730150</td>
      <td>0.916735</td>
      <td>7.800113</td>
    </tr>
    <tr>
      <th>2007-01-02</th>
      <td>8.521084</td>
      <td>8.512227</td>
      <td>0.720966</td>
      <td>7.800113</td>
    </tr>
    <tr>
      <th>2007-01-03</th>
      <td>8.367682</td>
      <td>8.385690</td>
      <td>0.567568</td>
      <td>7.800113</td>
    </tr>
    <tr>
      <th>2007-01-04</th>
      <td>8.386809</td>
      <td>8.394879</td>
      <td>0.586693</td>
      <td>7.800113</td>
    </tr>
    <tr>
      <th>2007-01-05</th>
      <td>8.418292</td>
      <td>8.409329</td>
      <td>0.618176</td>
      <td>7.800113</td>
    </tr>
  </tbody>
</table>
</div>




```python
for column in sites.columns:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(sites.index.to_timestamp(), sites[column], label=column)
    ax.set_title(column)
    fig.show()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[12]</span></p>


    
![png](index_files/output_18_0.png)
    



    
![png](index_files/output_18_1.png)
    



    
![png](index_files/output_18_2.png)
    



    
![png](index_files/output_18_3.png)
    


## Fitting with MCMC
In the previous examples, we used MAP inference to fit the model. However, we can also
use Markov Chain Monte Carlo (MCMC) to fit the model. To do this, we just need to
change the `inference_engine` parameter to `MCMCInferenceEngine`. The rest of the code
remains the same.

The `set_params` method is used to set the parameters of the model, in sklearn fashion.




```python
from prophetverse.engine import MCMCInferenceEngine

model.set_params(inference_engine=MCMCInferenceEngine())

import numpyro

numpyro.enable_x64()  # To avoid computational issues with MCMC

model.fit(y=y)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[13]</span></p>




<style>#sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 {
    /* Definition of color scheme common for light and dark mode */
    --sklearn-color-text: black;
    --sklearn-color-line: gray;
    /* Definition of color scheme for objects */
    --sklearn-color-level-0: #fff5e6;
    --sklearn-color-level-1: #f6e4d2;
    --sklearn-color-level-2: #ffe0b3;
    --sklearn-color-level-3: chocolate;

    /* Specific color for light theme */
    --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, white));
    --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-icon: #696969;

    @media (prefers-color-scheme: dark) {
      /* Redefinition of color scheme for dark theme */
      --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, #111));
      --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-icon: #878787;
    }
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 {
    color: var(--sklearn-color-text);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 pre {
    padding: 0;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 input.sk-hidden--visually {
    border: 0;
    clip: rect(1px 1px 1px 1px);
    clip: rect(1px, 1px, 1px, 1px);
    height: 1px;
    margin: -1px;
    overflow: hidden;
    padding: 0;
    position: absolute;
    width: 1px;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-text-repr-fallback {
    display: none;
  }

  div.sk-parallel-item,
  div.sk-serial,
  div.sk-item {
    /* draw centered vertical line to link estimators */
    background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
    background-size: 2px 100%;
    background-repeat: no-repeat;
    background-position: center center;
  }

  /* Parallel-specific style estimator block */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-serial {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--sklearn-color-background);
    padding-right: 1em;
    padding-left: 1em;
  }


  /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
  clickable and can be expanded/collapsed.
  - Pipeline and ColumnTransformer use this feature and define the default style
  - Estimators will overwrite some part of the style using the `sk-estimator` class
  */

  /* Pipeline and ColumnTransformer style (default) */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label label.sk-toggleable__label,
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 div.sk-estimator:hover {
    background-color: var(--sklearn-color-level-2);
  }

  /* Specification for estimator info */

  .sk-estimator-doc-link,
  a:link.sk-estimator-doc-link,
  a:visited.sk-estimator-doc-link {
    float: right;
    font-size: smaller;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1em;
    height: 1em;
    width: 1em;
    text-decoration: none !important;
    margin-left: 1ex;
    border: var(--sklearn-color-level-1) 1pt solid;
    color: var(--sklearn-color-level-1);
  }

  /* On hover */
  div.sk-estimator:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover,
  div.sk-label-container:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }

  /* Span, style for the box shown on hovering the info icon */
  .sk-estimator-doc-link span {
    display: none;
    z-index: 9999;
    position: relative;
    font-weight: normal;
    right: .2ex;
    padding: .5ex;
    margin: .5ex;
    width: min-content;
    min-width: 20ex;
    max-width: 50ex;
    color: var(--sklearn-color-text);
    box-shadow: 2pt 2pt 4pt #999;
    background: var(--sklearn-color-level-0);
    border: .5pt solid var(--sklearn-color-level-3);
  }

  .sk-estimator-doc-link:hover span {
    display: block;
  }

  /* "?"-specific style due to the `<a>` HTML tag */

  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 a.estimator_doc_link {
    float: right;
    font-size: 1rem;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1rem;
    height: 1rem;
    width: 1rem;
    text-decoration: none;
    color: var(--sklearn-color-level-1);
    border: var(--sklearn-color-level-1) 1pt solid;
  }

  /* On hover */
  #sk-c5902009-83e4-49a3-a5d5-1df7835d98a5 a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-c5902009-83e4-49a3-a5d5-1df7835d98a5' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MCMCInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-500))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('84d47860-a8bb-4bb4-ac69-bbaf48898c99') type="checkbox" ><label for=UUID('84d47860-a8bb-4bb4-ac69-bbaf48898c99') class='sk-toggleable__label sk-toggleable__label-arrow'>Prophetverse</label><div class="sk-toggleable__content"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MCMCInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=500,
                                        changepoint_prior_scale=1e-05,
                                        changepoint_range=-500))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>effects</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('cc88223e-9892-485a-8829-9d50798d83dc') type="checkbox" ><label for=UUID('cc88223e-9892-485a-8829-9d50798d83dc') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLinearTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLinearTrend(changepoint_interval=500, changepoint_prior_scale=1e-05,
                     changepoint_range=-500)</pre></div></div></div><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('8c5f5e65-c630-44e3-bee8-1c338ff3db37') type="checkbox" ><label for=UUID('8c5f5e65-c630-44e3-bee8-1c338ff3db37') class='sk-toggleable__label sk-toggleable__label-arrow'>LinearFourierSeasonality</label><div class="sk-toggleable__content"><pre>LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                         fourier_terms_list=[3, 10], freq=&#x27;D&#x27;, prior_scale=0.1,
                         sp_list=[7, 365.25])</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>inference_engine</label></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('fa1a3a4b-c44c-4dcd-be98-c1e90422d65d') type="checkbox" ><label for=UUID('fa1a3a4b-c44c-4dcd-be98-c1e90422d65d') class='sk-toggleable__label sk-toggleable__label-arrow'>MCMCInferenceEngine</label><div class="sk-toggleable__content"><pre>MCMCInferenceEngine()</pre></div></div></div></div></div></div></div></div></div></div>




```python
quantiles = model.predict_quantiles(fh=forecast_horizon, alpha=[0.75, 0.25])
fig, ax = plt.subplots(figsize=(10, 5))
# Plot area between quantiles
ax.fill_between(
    quantiles.index.to_timestamp(),
    quantiles.iloc[:, 0],
    quantiles.iloc[:, -1],
    alpha=0.5,
)
ax.scatter(y.index, y, marker="o", color="k", s=2, alpha=1)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[14]</span></p>




    <matplotlib.collections.PathCollection at 0x35575eb50>




    
![png](index_files/output_21_1.png)
    


One interesting feature of MCMC is that it allows us to obtain samples from the
posterior distribution of the parameters. In other words, we can also obtain probabilistic
forecasts for the TS components.



```python
samples = model.predict_component_samples(fh=forecast_horizon)
samples

```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[15]</span></p>




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>mean</th>
      <th>obs</th>
      <th>seasonality</th>
      <th>trend</th>
    </tr>
    <tr>
      <th>sample</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0</th>
      <th>2007-01-01</th>
      <td>13.216102</td>
      <td>13.196515</td>
      <td>4.843813</td>
      <td>8.372289</td>
    </tr>
    <tr>
      <th>2007-01-02</th>
      <td>13.454115</td>
      <td>13.701294</td>
      <td>5.081825</td>
      <td>8.372289</td>
    </tr>
    <tr>
      <th>2007-01-03</th>
      <td>11.383386</td>
      <td>11.329435</td>
      <td>3.011097</td>
      <td>8.372289</td>
    </tr>
    <tr>
      <th>2007-01-04</th>
      <td>10.875955</td>
      <td>10.969802</td>
      <td>2.503665</td>
      <td>8.372289</td>
    </tr>
    <tr>
      <th>2007-01-05</th>
      <td>12.315873</td>
      <td>12.678445</td>
      <td>3.943584</td>
      <td>8.372289</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">999</th>
      <th>2017-12-28</th>
      <td>7.777440</td>
      <td>7.705639</td>
      <td>0.620472</td>
      <td>7.156968</td>
    </tr>
    <tr>
      <th>2017-12-29</th>
      <td>8.144931</td>
      <td>8.526028</td>
      <td>0.988648</td>
      <td>7.156283</td>
    </tr>
    <tr>
      <th>2017-12-30</th>
      <td>7.878786</td>
      <td>7.964994</td>
      <td>0.723189</td>
      <td>7.155597</td>
    </tr>
    <tr>
      <th>2017-12-31</th>
      <td>7.821858</td>
      <td>7.957323</td>
      <td>0.666947</td>
      <td>7.154911</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>8.740103</td>
      <td>8.565384</td>
      <td>1.585878</td>
      <td>7.154225</td>
    </tr>
  </tbody>
</table>
<p>4019000 rows × 4 columns</p>
</div>



## Extra: syntax sugar
In Prophetverse, we've implemented the `>>` operator, which makes it easier to
set `trend`, `exogenous_effects` and `inference_engine` parameters.




```python

trend = PiecewiseLinearTrend(
    changepoint_interval=300,
    changepoint_prior_scale=0.0001,
    changepoint_range=0.8,
)
exogenous_effects = [
    (
        "seasonality",
        LinearFourierSeasonality(
            freq="D",
            sp_list=[7, 365.25],
            fourier_terms_list=[3, 10],
            prior_scale=0.1,
            effect_mode="multiplicative",
        ),
        no_input_columns,
    ),
]
inference_engine = MAPInferenceEngine()

model = Prophetverse() >> trend >> exogenous_effects >> inference_engine
model.fit(y=y)

```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[16]</span></p>




<style>#sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 {
    /* Definition of color scheme common for light and dark mode */
    --sklearn-color-text: black;
    --sklearn-color-line: gray;
    /* Definition of color scheme for objects */
    --sklearn-color-level-0: #fff5e6;
    --sklearn-color-level-1: #f6e4d2;
    --sklearn-color-level-2: #ffe0b3;
    --sklearn-color-level-3: chocolate;

    /* Specific color for light theme */
    --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, white));
    --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, black));
    --sklearn-color-icon: #696969;

    @media (prefers-color-scheme: dark) {
      /* Redefinition of color scheme for dark theme */
      --sklearn-color-text-on-default-background: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-background: var(--theme-background, var(--jp-layout-color0, #111));
      --sklearn-color-border-box: var(--theme-code-foreground, var(--jp-content-font-color1, white));
      --sklearn-color-icon: #878787;
    }
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 {
    color: var(--sklearn-color-text);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 pre {
    padding: 0;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 input.sk-hidden--visually {
    border: 0;
    clip: rect(1px 1px 1px 1px);
    clip: rect(1px, 1px, 1px, 1px);
    height: 1px;
    margin: -1px;
    overflow: hidden;
    padding: 0;
    position: absolute;
    width: 1px;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-text-repr-fallback {
    display: none;
  }

  div.sk-parallel-item,
  div.sk-serial,
  div.sk-item {
    /* draw centered vertical line to link estimators */
    background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
    background-size: 2px 100%;
    background-repeat: no-repeat;
    background-position: center center;
  }

  /* Parallel-specific style estimator block */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-serial {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--sklearn-color-background);
    padding-right: 1em;
    padding-left: 1em;
  }


  /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
  clickable and can be expanded/collapsed.
  - Pipeline and ColumnTransformer use this feature and define the default style
  - Estimators will overwrite some part of the style using the `sk-estimator` class
  */

  /* Pipeline and ColumnTransformer style (default) */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label label.sk-toggleable__label,
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 div.sk-estimator:hover {
    background-color: var(--sklearn-color-level-2);
  }

  /* Specification for estimator info */

  .sk-estimator-doc-link,
  a:link.sk-estimator-doc-link,
  a:visited.sk-estimator-doc-link {
    float: right;
    font-size: smaller;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1em;
    height: 1em;
    width: 1em;
    text-decoration: none !important;
    margin-left: 1ex;
    border: var(--sklearn-color-level-1) 1pt solid;
    color: var(--sklearn-color-level-1);
  }

  /* On hover */
  div.sk-estimator:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover,
  div.sk-label-container:hover .sk-estimator-doc-link:hover,
  .sk-estimator-doc-link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }

  /* Span, style for the box shown on hovering the info icon */
  .sk-estimator-doc-link span {
    display: none;
    z-index: 9999;
    position: relative;
    font-weight: normal;
    right: .2ex;
    padding: .5ex;
    margin: .5ex;
    width: min-content;
    min-width: 20ex;
    max-width: 50ex;
    color: var(--sklearn-color-text);
    box-shadow: 2pt 2pt 4pt #999;
    background: var(--sklearn-color-level-0);
    border: .5pt solid var(--sklearn-color-level-3);
  }

  .sk-estimator-doc-link:hover span {
    display: block;
  }

  /* "?"-specific style due to the `<a>` HTML tag */

  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 a.estimator_doc_link {
    float: right;
    font-size: 1rem;
    line-height: 1em;
    font-family: monospace;
    background-color: var(--sklearn-color-background);
    border-radius: 1rem;
    height: 1rem;
    width: 1rem;
    text-decoration: none;
    color: var(--sklearn-color-level-1);
    border: var(--sklearn-color-level-1) 1pt solid;
  }

  /* On hover */
  #sk-6500d34a-a1f7-4001-bc3e-86335b1688c8 a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-6500d34a-a1f7-4001-bc3e-86335b1688c8' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=300,
                                        changepoint_prior_scale=0.0001))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('bfd9ba43-e845-42ec-bffa-7ca2c713a1c9') type="checkbox" ><label for=UUID('bfd9ba43-e845-42ec-bffa-7ca2c713a1c9') class='sk-toggleable__label sk-toggleable__label-arrow'>Prophetverse</label><div class="sk-toggleable__content"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[3,
                                                                              10],
                                                          freq=&#x27;D&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[7, 365.25]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLinearTrend(changepoint_interval=300,
                                        changepoint_prior_scale=0.0001))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>effects</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('5c728c02-4b3a-424d-b694-5778adef50b2') type="checkbox" ><label for=UUID('5c728c02-4b3a-424d-b694-5778adef50b2') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLinearTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLinearTrend(changepoint_interval=300, changepoint_prior_scale=0.0001)</pre></div></div></div><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('26334aab-cea3-4fb7-a0a7-1e1799697821') type="checkbox" ><label for=UUID('26334aab-cea3-4fb7-a0a7-1e1799697821') class='sk-toggleable__label sk-toggleable__label-arrow'>LinearFourierSeasonality</label><div class="sk-toggleable__content"><pre>LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                         fourier_terms_list=[3, 10], freq=&#x27;D&#x27;, prior_scale=0.1,
                         sp_list=[7, 365.25])</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><label>inference_engine</label></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('8894fdb6-5f4b-4e48-9cd0-29acd13b938b') type="checkbox" ><label for=UUID('8894fdb6-5f4b-4e48-9cd0-29acd13b938b') class='sk-toggleable__label sk-toggleable__label-arrow'>MAPInferenceEngine</label><div class="sk-toggleable__content"><pre>MAPInferenceEngine()</pre></div></div></div></div></div></div></div></div></div></div>




```python
forecast_horizon = pd.period_range("2007-01-01", "2018-01-01", freq="D")
fig, ax = plt.subplots(figsize=(10, 5))
preds = model.predict(fh=forecast_horizon)
preds.plot.line(ax=ax)
ax.scatter(y.index, y, marker="o", color="k", s=2, alpha=0.5)

```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[17]</span></p>




    <matplotlib.collections.PathCollection at 0x36001d2d0>




    
![png](index_files/output_26_1.png)
    

