# Hierarchical Prophet
<p align="center">
<img src="docs/static/logo.webp" width="200">

</p>


This library was created to make a numpyro-based Prophet model for timeseries forecasting. In addition, it also provides an extension that allows for hierarchical forecasting simultaneously, with potentially shared coefficients between timeseries. All implementations (hierarchical and univariate) are based on sktime interface.

The hierarchical one creates a Prophet-like model for each bottom series, but uses a multivariate normal likelihood based on the hierarchy structure to also consider information of middle-top levels.


## Installation

To install with pip:

```bash
pip install hierarchical-prophet
```

Or with poetry:

```bash
poetry add hierarchical-prophet
```



## Hierarchical Model

Let $Y_{bottom} \in \mathcal{R}^{b}$ be the random vector of bottom series which follow a normal distribution $\mathcal{N}(y_{bottom}, \Sigma)$, where $\Sigma \in \mathcal{R}^{b \times b}$ is the covariance matrix of the bottom series which we assume is diagonal. In addition, let $ S \in \mathcal{R}^{m \times b}$ be the matrix that define the hierarchical structure of the series, where $m \geq b$ is the total number of series . Then, the random variable $Y$ which define the value of all series is defined as:

$$
Y = SY_{bottom}
$$

Its distribution is given by:

$$
Y  \sim \mathcal{N}(SY_{bottom}, S\Sigma S^T)
$$

A custom distribution was implemented so that samples are drawn according to that multivariate distribution - in a bottom-up fashion -, and the likelihood applied to all levels at once.


# Installation



# Remarks

## Differences between this Prophet and the original one

The main differences with the original Prophet model are:

1. The logistic version of the trend. In the original paper, the logistic growth is:

    $$
    trend = \frac{C}{1 + \exp(-k(t - m))}
    $$

    where $C$ is the capacity, $k$ is the growth rate and $m$ is the inflection point. In this implementation, we implement a similar and equivalent model, but with a different parameterization:

    $$
    trend = \frac{C}{1 + \exp(-(kt + m))}
    $$

    which are equivalent. The priors for those parameters $k$ and $m$ are chosen in a data driven way, so that they match the maximum and minimum value of the series.

2. The capacity is also modelled as a random variable, and it's assumed constant.
3. One can set different prior distributions for the parameters of the model. The parameters also may be different for different groups of variables, which allows to force positive coefficients for some groups and not for others (with HalfNormal prior, for example).
4. The exogenous variable inputs are not scaled. They should be scaled prior to the model fitting, with sktime transfomers for example.


### Differences between this Hierarchical Prophet and the original one

1. All the above
2. Changepoint interval is used instead of changepoint number. Motivation: as the timeseries evolve, a given changepoint number may have different meanings. For example, a changepoint number of 10 may be too much for a series with 100 observations, but too little for a series with 1000 observations. The changepoint interval may avoid this problem.
3. The fourier terms for seasonality must be passed as exogenous variables.
For the moment, the following features are not implemented:

- Maximum a posteriori estimation (mcmc_samples=0). Only MCMC with NUTS is implemented.
- There's no method/function for extracting the timeseries components (trend, seasonality, etc) from the model directly. Although 100% possible, it's not implemented as a function or method yet. Contributions are welcome!
