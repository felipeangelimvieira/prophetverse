# Prophetverse

<p align="center">
<img src="static/logo-removebg.png" width="300">
</p>


This library was created to make a numpyro-based Prophet model for timeseries forecasting. In addition, it also provides an extension that allows for hierarchical forecasting simultaneously, with potentially shared coefficients between timeseries. All implementations (hierarchical and univariate) are based on sktime interface.

The idea was not to fully reproduce Prophet, but to provide an extension where the capacity is a random variable, and the hierarchical structure is considered. The hierarchical one creates a Prophet-like model for each bottom series, but uses a multivariate normal likelihood based on the hierarchy structure to also consider information of middle-top levels.


## Installation

To install with pip:

```bash
pip install prophetverse
```

Or with poetry:

```bash
poetry add prophetverse
```



# Remarks

## Differences between this Prophet and the original one

The main differences with the original Prophet model are:

1. The logistic version of the trend. In the original paper, the logistic growth is:

    $$
    trend = \frac{C}{1 + \exp(-k(t - m))}
    $$

    where $C$ is the capacity, $k$ is the growth rate and $m$ is the timeoffset. In this implementation, we implement a similar and equivalent model, but with a different parameterization:

    $$
    trend = \frac{C}{1 + \exp(-(kt + m'))}
    $$

    which are equivalent. The priors for those parameters $k$ and $m$ are chosen in a data driven way, so that they match the maximum and minimum value of the series.

2. The capacity is also modelled as a random variable, and it's assumed constant.
3. One can set different prior distributions for the parameters of the model. The parameters also may be different for different groups of variables, which allows to force positive coefficients for some groups and not for others (with HalfNormal prior, for example).
4. Changepoint interval is used instead of changepoint number. Motivation: as the timeseries evolve, a given changepoint number may have different meanings. For example, a changepoint number of 10 may be too much for a series with 100 observations, but too little for a series with 1000 observations. The changepoint interval may avoid this problem.
5. The exogenous variable inputs are not scaled. They should be scaled prior to the model fitting, with sktime transfomers for example.
6. The fourier terms for seasonality must be passed as exogenous variables in `feature_transformer` argument.

For the hierarchical model, the forecast is done in a bottom-up fashion. All series parameters are infered simultaneously, and a multivariate normal likelihood is used (LKJ prior for the correlation matrix). In the future, forecasts with OLS reconciliation may be implemented.
