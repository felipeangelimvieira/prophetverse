# Prophetverse

<p align="center">
<img src="static/logo-removebg.png" width="200">

</p>

[![PyPI version](https://badge.fury.io/py/prophetverse.svg)](https://badge.fury.io/py/prophetverse)
[![codecov](https://codecov.io/gh/felipeangelimvieira/prophetverse/graph/badge.svg?token=O37PGJI3ZX)](https://codecov.io/gh/felipeangelimvieira/prophetverse)

Prophetverse leverages the theory behind the Prophet model for time series forecasting and expands it into __a more general framework__, enabling custom priors, non-linear effects for exogenous variables and other likelihoods. Built on top of [sktime](https://www.sktime.net/en/stable/) and [numpyro](https://num.pyro.ai/en/stable/), Prophetverse aims to provide a flexible and easy-to-use library for time series forecasting with a focus on interpretability and customizability. It is particularly useful for Marketing Mix Modeling, where understanding the effect of different marketing channels on sales is crucial.


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    Install `Prophetverse` with `pip` and get up
    and forecasting in minutes

    [:octicons-arrow-right-24: Getting started](getting-started)

-   :material-school:{ .lg .middle } __Theory__

    ---

    Understand the idea behind the model and how it works

    [:octicons-arrow-right-24: Read the post](the-theory)

-   :material-file:{ .lg .middle } __Examples__

    ---

    Go the example gallery and see how to use the model, and its features!

    [:octicons-arrow-right-24: Examples](examples/univariate)

-   :material-book:{ .lg .middle } __Reference__

    ---

    Take a look at the API Reference to see all the available options

    [:octicons-arrow-right-24: Reference](reference/sktime/prophetverse/)


</div>




### Features

Prophetverse is similar to the original Prophet model in many aspects, but it has some differences and new features. The following table summarizes the main features of Prophetverse and compares them with the original Prophet model:



| Feature                         | Prophetverse                                                                                                               | Original Prophet                         | Motivation |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------| ----------------------------------------- |
| **Logistic trend**              | Capacity as a random variable                           | Capacity as a hyperparameter, user input required             | The capacity is usually unknown by the users. Having it as a variable is useful for Total Addressable Market inference |
| **Custom trend**               | Customizable trend functions                                                                                                | Not available                            | Users can create custom trends and leverage their knowledge about the timeseries to enhance long-term accuracy |
| **Likelihoods**                 | Gaussian, Gamma and Negative Binomial                                                                                                | Gaussian only                            | Gaussian likelihood fails to provide good forecasts to positive-only and count data (sales, for example) |
| **Custom priors**               | Supports custom priors for model parameters and exogenous variables                                                        | Not supported                            | Forcing positive coefficients, using prior knowledge to model the timeseries|
| **Custom exogenous effects**              | Non-linear and customizable effects for exogenous variables, shared coefficients between time series                       | Not available                            | Users can create any kind of relationship between exogenous variables and the timeseries, which can be useful for Marketing Mix Modeling and other applications. |
| **Changepoints**                | Uses changepoint interval                                                                                                  | Uses changepoint number                  | The changepoint number is not stable in the sense that, when the size of timeseries increases, its impact on forecast changes. Think about setting a changepoint number when timeseries has 6 months, and forecasting in future with 2 years of data (4x time original size). Re-tuning would be required. Prophetverse is expected to be more stable |
| **Scaling**                     | Time series scaled internally, exogenous variables scaled by the user                                                      | Time series scaled internally            | Scaling `y` is needed to enhance user experience with hyperparameters. On the other hand, not scaling the exogenous variables provide more control to the user and they can leverage sktime's transformers to handle that. |
| **Seasonality**                 | Fourier terms for seasonality passed as exogenous variables                                                                | Built-in seasonality handling            | Setting up seasonality requires almost zero effort by using `LinearFourierSeasonality` in Prophetverse. The idea is to allow the user to create custom seasonalities easily, without hardcoding it in the code. |
| **Multivariate model**          | Hierarchical model with multivariate normal likelihood and LKJ prior, bottom-up forecast                                    | Not available                            | Having shared coefficients, using global information to enhance individual forecast.|| **Inference methods**           | MCMC and MAP                                                                                                               | MCMC and MAP                            | |
| **Implementation** | Numpyro | Stan
