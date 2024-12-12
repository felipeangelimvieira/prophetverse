# Hierarchical timeseries
In this example, we will show how to forecast hierarchical timeseries
with the univariate `Prophetverse` and `HierarchicalProphet` models

The univariate `Prophetverse` model can seamlessly handle hierarchical timeseries
due to the package's compatibility with sktime. The `HierarchicalProphet` model
is specifically designed to handle hierarchical timeseries, by forecasting all
bottom-level series at once.

!!! note
    Currently, some features of the univariate Prophet are not available in the hierarchical
    version, such as likelihoods different from Gaussian. We are looking forward to
    adding these features in the future.





```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophetverse.datasets.loaders import load_tourism


```

## Import dataset

Here we use the tourism dataset with purpose-level aggregation.



```python


y = load_tourism(groupby="Purpose")
display(y)


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
      <th></th>
      <th>Trips</th>
    </tr>
    <tr>
      <th>Purpose</th>
      <th>Quarter</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Business</th>
      <th>1998Q1</th>
      <td>7391.962068</td>
    </tr>
    <tr>
      <th>1998Q2</th>
      <td>7701.153191</td>
    </tr>
    <tr>
      <th>1998Q3</th>
      <td>8911.852065</td>
    </tr>
    <tr>
      <th>1998Q4</th>
      <td>7777.766525</td>
    </tr>
    <tr>
      <th>1999Q1</th>
      <td>6917.257864</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">__total</th>
      <th>2015Q4</th>
      <td>51518.858354</td>
    </tr>
    <tr>
      <th>2016Q1</th>
      <td>54984.720748</td>
    </tr>
    <tr>
      <th>2016Q2</th>
      <td>49583.595515</td>
    </tr>
    <tr>
      <th>2016Q3</th>
      <td>49392.159616</td>
    </tr>
    <tr>
      <th>2016Q4</th>
      <td>54034.155613</td>
    </tr>
  </tbody>
</table>
<p>380 rows × 1 columns</p>
</div>



We define the helper function below to plot the predictions and the observations.




```python
LEVELS = y.index.get_level_values(0).unique()


def plot_preds(y=None, preds={}, axs=None):

    if axs is None:
        fig, axs = plt.subplots(
            figsize=(12, 8), nrows=int(np.ceil(len(LEVELS) / 2)), ncols=2
        )
    ax_generator = iter(axs.flatten())
    for level in LEVELS:
        ax = next(ax_generator)
        if y is not None:
            y.loc[level].iloc[:, 0].rename("Observation").plot(
                ax=ax, label="truth", color="black"
            )
        for name, _preds in preds.items():
            _preds.loc[level].iloc[:, 0].rename(name).plot(ax=ax, legend=True)
        ax.set_title(level)

    # Tight layout
    plt.tight_layout()
    return ax



```

## Fit univariate model

Because of sktime's amazing interface, we can use the univariate Prophet seamlessly with hierarchical data. We do not reconcile it here, but it could be achieved with the `Reconciler` class.





```python

import jax.numpy as jnp

from prophetverse.effects import LinearFourierSeasonality
from prophetverse.effects.trend import (PiecewiseLinearTrend,
                                        PiecewiseLogisticTrend)
from prophetverse.engine import MAPInferenceEngine
from prophetverse.sktime.univariate import Prophetverse
from prophetverse.utils import no_input_columns

model = Prophetverse(
    trend=PiecewiseLogisticTrend(
        changepoint_prior_scale=0.1,
        changepoint_interval=8,
        changepoint_range=-8,
    ),
    exogenous_effects=[
        (
            "seasonality",
            LinearFourierSeasonality(
                sp_list=["Y"],
                fourier_terms_list=[1],
                freq="Q",
                prior_scale=0.1,
                effect_mode="multiplicative",
            ),
            no_input_columns,
        )
    ],
    inference_engine=MAPInferenceEngine()
)
model.fit(y=y)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[5]</span></p>




<style>#sk-adfa5277-59a6-42da-8920-9915af87216e {
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

  #sk-adfa5277-59a6-42da-8920-9915af87216e {
    color: var(--sklearn-color-text);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e pre {
    padding: 0;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e input.sk-hidden--visually {
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

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-text-repr-fallback {
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

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-serial {
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

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-adfa5277-59a6-42da-8920-9915af87216e label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label label.sk-toggleable__label,
  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-adfa5277-59a6-42da-8920-9915af87216e div.sk-estimator:hover {
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

  #sk-adfa5277-59a6-42da-8920-9915af87216e a.estimator_doc_link {
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
  #sk-adfa5277-59a6-42da-8920-9915af87216e a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-adfa5277-59a6-42da-8920-9915af87216e' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[1],
                                                          freq=&#x27;Q&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[&#x27;Y&#x27;]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x1574ef510&gt;,
                                          changepoint_interval=8,
                                          changepoint_prior_scale=0.1,
                                          changepoint_range=-8))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('76cb1c0d-9f2c-4137-8026-18f532b3cb72') type="checkbox" ><label for=UUID('76cb1c0d-9f2c-4137-8026-18f532b3cb72') class='sk-toggleable__label sk-toggleable__label-arrow'>Prophetverse</label><div class="sk-toggleable__content"><pre>Prophetverse(exogenous_effects=[(&#x27;seasonality&#x27;,
                                 LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                          fourier_terms_list=[1],
                                                          freq=&#x27;Q&#x27;,
                                                          prior_scale=0.1,
                                                          sp_list=[&#x27;Y&#x27;]),
                                 &#x27;^$&#x27;)],
             inference_engine=MAPInferenceEngine(),
             trend=PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x1574ef510&gt;,
                                          changepoint_interval=8,
                                          changepoint_prior_scale=0.1,
                                          changepoint_range=-8))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('b2285921-3b43-422f-afe4-574a4d7d2f30') type="checkbox" ><label for=UUID('b2285921-3b43-422f-afe4-574a4d7d2f30') class='sk-toggleable__label sk-toggleable__label-arrow'>trend: PiecewiseLogisticTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x1574ef510&gt;,
                       changepoint_interval=8, changepoint_prior_scale=0.1,
                       changepoint_range=-8)</pre></div></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('7c0ad06c-97a3-4abf-aff9-038ac780e8a8') type="checkbox" ><label for=UUID('7c0ad06c-97a3-4abf-aff9-038ac780e8a8') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLogisticTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x1574ef510&gt;,
                       changepoint_interval=8, changepoint_prior_scale=0.1,
                       changepoint_range=-8)</pre></div></div></div></div></div></div></div></div></div></div>



### Forecasting with automatic upcasting
To call the same methods we used in the univariate case, we do not need to change
a single line of code. The only difference is that the output will be a `pd.DataFrame`
with more rows and index levels.




```python
forecast_horizon = pd.period_range("1997Q1",
                                   "2020Q4",
                                   freq="Q")
preds = model.predict(fh=forecast_horizon)
display(preds.head())

# Plot
plot_preds(y, {"Prophet": preds})
plt.show()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[6]</span></p>


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
      <th>Trips</th>
    </tr>
    <tr>
      <th>Purpose</th>
      <th>Quarter</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Business</th>
      <th>1997Q1</th>
      <td>7091.107910</td>
    </tr>
    <tr>
      <th>1997Q2</th>
      <td>8024.383301</td>
    </tr>
    <tr>
      <th>1997Q3</th>
      <td>8942.321289</td>
    </tr>
    <tr>
      <th>1997Q4</th>
      <td>8000.666992</td>
    </tr>
    <tr>
      <th>1998Q1</th>
      <td>7091.107910</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](index_files/output_9_1.png)
    


The same applies to the decomposition method:




```python
decomposition = model.predict_components(fh=forecast_horizon)
decomposition.head()

```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[7]</span></p>




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
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Business</th>
      <th>1997Q1</th>
      <td>7091.107910</td>
      <td>7066.821777</td>
      <td>-925.401123</td>
      <td>8016.504883</td>
    </tr>
    <tr>
      <th>1997Q2</th>
      <td>8024.383301</td>
      <td>8028.772461</td>
      <td>7.871942</td>
      <td>8016.504883</td>
    </tr>
    <tr>
      <th>1997Q3</th>
      <td>8942.321289</td>
      <td>8965.629883</td>
      <td>925.810303</td>
      <td>8016.504883</td>
    </tr>
    <tr>
      <th>1997Q4</th>
      <td>8000.666992</td>
      <td>8041.334473</td>
      <td>-15.839881</td>
      <td>8016.504883</td>
    </tr>
    <tr>
      <th>1998Q1</th>
      <td>7091.107910</td>
      <td>7084.361816</td>
      <td>-925.401123</td>
      <td>8016.504883</td>
    </tr>
  </tbody>
</table>
</div>



## Hierarchical Prophet

Now, let's use the hierarchical prophet to forecast all of the series at once.
The interface here is the same as the univariate case. The fit step can
take a little longer since there are more parameters to estimate.




```python

from prophetverse.logger import logger

# Set debug level everywhere
logger.setLevel("DEBUG")
logger = logger.getChild("lbfgs")
logger.setLevel("DEBUG")
import numpyro

from prophetverse.sktime.multivariate import HierarchicalProphet

numpyro.enable_x64()
model_hier = HierarchicalProphet(
    trend=PiecewiseLogisticTrend(
        changepoint_prior_scale=0.1,
        changepoint_interval=8,
        changepoint_range=-8,
    ),
    exogenous_effects=[
        (
            "seasonality",
            LinearFourierSeasonality(
                sp_list=["Y"],
                fourier_terms_list=[1],
                freq="Q",
                prior_scale=0.1,
                effect_mode="multiplicative",
            ),
            no_input_columns,
        )
    ],
    inference_engine=MAPInferenceEngine(),
)


model_hier.fit(y=y)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[8]</span></p>




<style>#sk-35a15836-7360-49d5-84d9-f4fdf785d726 {
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

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 {
    color: var(--sklearn-color-text);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 pre {
    padding: 0;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 input.sk-hidden--visually {
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

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-dashed-wrapped {
    border: 1px dashed var(--sklearn-color-line);
    margin: 0 0.4em 0.5em 0.4em;
    box-sizing: border-box;
    padding-bottom: 0.4em;
    background-color: var(--sklearn-color-background);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-container {
    /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
       but bootstrap.min.css set `[hidden] { display: none !important; }`
       so we also need the `!important` here to be able to override the
       default hidden behavior on the sphinx rendered scikit-learn.org.
       See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
    display: inline-block !important;
    position: relative;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-text-repr-fallback {
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

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel-item::after {
    content: "";
    width: 100%;
    border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
    flex-grow: 1;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel {
    display: flex;
    align-items: stretch;
    justify-content: center;
    background-color: var(--sklearn-color-background);
    position: relative;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel-item {
    display: flex;
    flex-direction: column;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel-item:first-child::after {
    align-self: flex-end;
    width: 50%;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel-item:last-child::after {
    align-self: flex-start;
    width: 50%;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-parallel-item:only-child::after {
    width: 0;
  }

  /* Serial-specific style estimator block */

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-serial {
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

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-toggleable {
    /* Default theme specific background. It is overwritten whether we have a
    specific estimator or a Pipeline/ColumnTransformer */
    background-color: var(--sklearn-color-background);
  }

  /* Toggleable label */
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 label.sk-toggleable__label {
    cursor: pointer;
    display: block;
    width: 100%;
    margin-bottom: 0;
    padding: 0.5em;
    box-sizing: border-box;
    text-align: center;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 label.sk-toggleable__label-arrow:before {
    /* Arrow on the left of the label */
    content: "▸";
    float: left;
    margin-right: 0.25em;
    color: var(--sklearn-color-icon);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 label.sk-toggleable__label-arrow:hover:before {
    color: var(--sklearn-color-text);
  }

  /* Toggleable content - dropdown */

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-toggleable__content {
    max-height: 0;
    max-width: 0;
    overflow: hidden;
    text-align: left;
    background-color: var(--sklearn-color-level-0);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-toggleable__content pre {
    margin: 0.2em;
    border-radius: 0.25em;
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-0);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 input.sk-toggleable__control:checked~div.sk-toggleable__content {
    /* Expand drop-down */
    max-height: 200px;
    max-width: 100%;
    overflow: auto;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
    content: "▾";
  }

  /* Pipeline/ColumnTransformer-specific style */

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator-specific style */

  /* Colorize estimator box */
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
    /* unfitted */
    background-color: var(--sklearn-color-level-2);
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label label.sk-toggleable__label,
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label label {
    /* The background is the default theme color */
    color: var(--sklearn-color-text-on-default-background);
  }

  /* On hover, darken the color of the background */
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label:hover label.sk-toggleable__label {
    color: var(--sklearn-color-text);
    background-color: var(--sklearn-color-level-2);
  }

  /* Estimator label */

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label label {
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    line-height: 1.2em;
  }

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-label-container {
    text-align: center;
  }

  /* Estimator-specific */
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-estimator {
    font-family: monospace;
    border: 1px dotted var(--sklearn-color-border-box);
    border-radius: 0.25em;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    background-color: var(--sklearn-color-level-0);
  }

  /* on hover */
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 div.sk-estimator:hover {
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

  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 a.estimator_doc_link {
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
  #sk-35a15836-7360-49d5-84d9-f4fdf785d726 a.estimator_doc_link:hover {
    background-color: var(--sklearn-color-level-3);
    color: var(--sklearn-color-background);
    text-decoration: none;
  }
</style><div id='sk-35a15836-7360-49d5-84d9-f4fdf785d726' class="sk-top-container"><div class="sk-text-repr-fallback"><pre>HierarchicalProphet(exogenous_effects=[(&#x27;seasonality&#x27;,
                                        LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                                 fourier_terms_list=[1],
                                                                 freq=&#x27;Q&#x27;,
                                                                 prior_scale=0.1,
                                                                 sp_list=[&#x27;Y&#x27;]),
                                        &#x27;^$&#x27;)],
                    inference_engine=MAPInferenceEngine(),
                    trend=PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x31706b790&gt;,
                                                 changepoint_interval=8,
                                                 changepoint_prior_scale=0.1,
                                                 changepoint_range=-8))</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('fc3a63c6-e4a5-4bbb-89ba-9c7961695d48') type="checkbox" ><label for=UUID('fc3a63c6-e4a5-4bbb-89ba-9c7961695d48') class='sk-toggleable__label sk-toggleable__label-arrow'>HierarchicalProphet</label><div class="sk-toggleable__content"><pre>HierarchicalProphet(exogenous_effects=[(&#x27;seasonality&#x27;,
                                        LinearFourierSeasonality(effect_mode=&#x27;multiplicative&#x27;,
                                                                 fourier_terms_list=[1],
                                                                 freq=&#x27;Q&#x27;,
                                                                 prior_scale=0.1,
                                                                 sp_list=[&#x27;Y&#x27;]),
                                        &#x27;^$&#x27;)],
                    inference_engine=MAPInferenceEngine(),
                    trend=PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x31706b790&gt;,
                                                 changepoint_interval=8,
                                                 changepoint_prior_scale=0.1,
                                                 changepoint_range=-8))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class='sk-label-container'><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('1b928981-e680-4ef0-9f9f-e118db92d1e1') type="checkbox" ><label for=UUID('1b928981-e680-4ef0-9f9f-e118db92d1e1') class='sk-toggleable__label sk-toggleable__label-arrow'>trend: PiecewiseLogisticTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x31706b790&gt;,
                       changepoint_interval=8, changepoint_prior_scale=0.1,
                       changepoint_range=-8)</pre></div></div></div><div class="sk-serial"><div class='sk-item'><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id=UUID('277bb426-2088-4ab8-b116-1c87af6e64cb') type="checkbox" ><label for=UUID('277bb426-2088-4ab8-b116-1c87af6e64cb') class='sk-toggleable__label sk-toggleable__label-arrow'>PiecewiseLogisticTrend</label><div class="sk-toggleable__content"><pre>PiecewiseLogisticTrend(capacity_prior=&lt;numpyro.distributions.distribution.TransformedDistribution object at 0x31706b790&gt;,
                       changepoint_interval=8, changepoint_prior_scale=0.1,
                       changepoint_range=-8)</pre></div></div></div></div></div></div></div></div></div></div>



### Forecasting with hierarchical prophet




```python
preds_hier = model_hier.predict(fh=forecast_horizon)

plot_preds(
    y,
    preds={
        "Prophet": preds,
        "HierarchicalProphet": preds_hier,
    },
)


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[9]</span></p>




    <Axes: title={'center': '__total'}, xlabel='Quarter'>




    
![png](index_files/output_15_1.png)
    


An important difference between the probabilistic features of the
univariate and hierarchical models is that the latter returns quantiles which
consider the correlation between the series. The samples used to compute such quantiles
come from reconciled predictive distributions.





```python
quantiles = model_hier.predict_quantiles(fh=forecast_horizon,
                                         alpha=[0.05, 0.95])
quantiles



```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[10]</span></p>




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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Trips</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>0.05</th>
      <th>0.95</th>
    </tr>
    <tr>
      <th>Purpose</th>
      <th>Quarter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Business</th>
      <th>1997Q1</th>
      <td>6391.811183</td>
      <td>7893.525922</td>
    </tr>
    <tr>
      <th>1997Q2</th>
      <td>7278.085487</td>
      <td>8801.051143</td>
    </tr>
    <tr>
      <th>1997Q3</th>
      <td>8222.493302</td>
      <td>9755.252573</td>
    </tr>
    <tr>
      <th>1997Q4</th>
      <td>7295.384599</td>
      <td>8802.309692</td>
    </tr>
    <tr>
      <th>1998Q1</th>
      <td>6419.552910</td>
      <td>7911.289437</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">__total</th>
      <th>2019Q4</th>
      <td>56023.305365</td>
      <td>60224.552213</td>
    </tr>
    <tr>
      <th>2020Q1</th>
      <td>59080.031026</td>
      <td>63381.339848</td>
    </tr>
    <tr>
      <th>2020Q2</th>
      <td>55916.642850</td>
      <td>60144.129946</td>
    </tr>
    <tr>
      <th>2020Q3</th>
      <td>53261.024309</td>
      <td>57668.196991</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>57156.985932</td>
      <td>61525.298753</td>
    </tr>
  </tbody>
</table>
<p>480 rows × 2 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(10, 5))

selected_series = "__total"
series = quantiles.loc[selected_series]
ax.fill_between(
    series.index.to_timestamp(),
    series.iloc[:, 0],
    series.iloc[:, -1],
    alpha=0.5,
)
ax.scatter(y.loc[selected_series].index, y.loc[selected_series], marker="o", color="k", alpha=1)
fig.show()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[11]</span></p>


    
![png](index_files/output_18_0.png)
    


### Decomposition with hierarchical prophet
We can also extract the components of the time series with the `predict_components`




```python
from sktime.transformations.hierarchical.aggregate import Aggregator

sites = model_hier.predict_components(fh=forecast_horizon)
sites = Aggregator(flatten_single_levels=True).fit_transform(sites)

for column in sites.columns.difference(["obs"]):
    fig, axs = plt.subplots(
        figsize=(12, 8), nrows=int(np.ceil(len(LEVELS) / 2)), ncols=2
    )
    plot_preds(preds={column: sites[[column]]}, axs=axs)
    # Set figure title
    fig.suptitle(column)
    fig.tight_layout()


```
<p class="cell-output-title jp-RenderedText jp-OutputArea-output">Output: <span class="cell-output-count">[12]</span></p>


    
![png](index_files/output_20_0.png)
    



    
![png](index_files/output_20_1.png)
    



    
![png](index_files/output_20_2.png)
    

