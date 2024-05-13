# Prophetverse

<p align="center">
<img src="docs/static/logo-removebg.png" width="200">

</p>

[![PyPI version](https://badge.fury.io/py/prophetverse.svg)](https://badge.fury.io/py/prophetverse)
[![codecov](https://codecov.io/gh/felipeangelimvieira/prophetverse/graph/badge.svg?token=O37PGJI3ZX)](https://codecov.io/gh/felipeangelimvieira/prophetverse)


This library was created to provide new versions of the Prophet model for timeseries forecasting. Implemented with numpyro, it allows to provide custom priors for specific groups of exogenous variables, and also offers a multivariate implementation for hierarchical forecasting, with potentially shared coefficients between timeseries. All implementations are based on sktime interface.


### Features

✅ Univariate and multivariate forecasting

✅ Gamma-likelihood and Negative Binomial likelihood for count data

✅ Custom prior distributions for exogenous variables

✅ Non-linear effects for exogenous variables (one may create custom effects inheriting AbstractEffect class)

✅ Shared coefficients between timeseries (multi-variate model)

✅ Sktime interface

✅ Capacity parameter of logistic trend as a random variable

✅ MCMC and MAP inference


## Installation

To install with pip:

```bash
pip install prophetverse
```

Or with poetry:

```bash
poetry add prophetverse
```

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
