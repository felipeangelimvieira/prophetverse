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




✅ Univariate and multivariate forecasting

📊 Gamma-likelihood and Negative Binomial likelihood for count data

🎲 Custom prior distributions for exogenous variables

🎯 Non-linear and customizable effects for exogenous variables

🔗 Shared coefficients between timeseries (multi-variate model)

🌐 Sktime interface

📈 Capacity parameter of logistic trend as a random variable, and customizable trends.

✨ MCMC and MAP inference


## Differences between this Prophet and the original one

The main differences with the original Prophet model are:

### Logistic trend

In this implementation, the capacity is modeled as a random variable and is assumed to be constant. In the original model, it was necessary to pass the capacity as a hyperparameter, but we often don't know the maximum value. One example is forecasting the number of new users of a product. We may not know surely what the maximum number of new users is, and may be particularly interested in it.

### Gamma and Negative Binomial likelihoods

The original model only supports Gaussian likelihood. This implementation supports Gamma and Negative Binomial likelihoods, which are useful for count data. 

### Custom priors

Users can set different prior distributions for the model parameters and define custom relationships between the exogenous variables and their effects on the mean. For example, one may want to force a positive effect of a variable on the mean, and use a HalfNormal prior for the coefficient and a `prophetverse.effects.LinearEffect` for the effect (see examples for more details).

I believe this is one of the most important features of this library. It opens the door to a lot of applications, such as Marketing Mix Modeling, which has the objective of understanding the effect of different marketing channels on sales. A saturating effect, such as a Hill Function, can be used to model the diminishing returns of a given channel.

### Changepoints

The changepoint interval is used instead of the changepoint number. Motivation: as the time series evolve, a given changepoint number may have different meanings. For example, a changepoint number of 10 may be too much for a series with 100 observations but too little for a series with 1000 observations. The changepoint interval may avoid this problem and avoid the need of tuning this hyperparameter frequently.

### Scaling

The time series is scaled internally as it is in the original model to provide more stable hyperparameters. However, exogenous variables must be scaled by the user. For that, you can use sktime's transformers and pass them to the `feature_transformer` argument of the model. 

### Seasonality

The Fourier terms for seasonality must be passed as exogenous variables in the `feature_transformer` argument, see [FourierFeatures](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.series.fourier.FourierFeatures.html) for a ready-to-use transformer. Also, check the examples in this documentation.

### Multivariate model

For the hierarchical model, the forecast is done in a bottom-up fashion. All series parameters are inferred simultaneously, and a multivariate normal likelihood is used (with LKJ prior for the correlation matrix). In the future, forecasts with OLS reconciliation may be implemented.

This model is also useful if you want to share coefficients of exogenous variables between time series. For example, if you have a dataset with multiple time series of sales of different products, you may want to share the effect of a given marketing channel between them. This is also possible with this implementation.
