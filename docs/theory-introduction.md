# Prophetverse (Pverse)

<p align="center">
<img src="/static/prophetverse-universe.png">
</p>

<p align="center" style="font-size: smaller;">
Figure 1: Generalized Additive Models are versatile. Pverse is one of the many models that can be built on top of it.
The idea of Pverse is giving access to that universe.
</p>

## Introduction

### Generalized Linear Models (GLM)

__Prophetverse__(Pverse) leverages the concept of the __Generalized Additive Model (GAM)__ to model time series data. The core principle of GAMs is to model the expected value $y_{mean}$ of 
the endogenous variable $Y$ as the sum of many functions $\{f_i\}_{i=1}^n$ of exogenous
variables $\{x_i\}_{i=1}^n$. 

$$
y_{mean} = f_1(x_1) + f_2(x_2) + \ldots + f_n(t)\text{, }\quad n \in \mathbb{N}
$$

In Pverse some of these functions can represent  trend, and seasonality similar to the [classical time series decomposition](https://otexts.com/fpp3/decomposition.html). 

### Linear Trend

The linear trend in Pverse is modeled using piecewise linear functions with changepoints. Each change point represents a discontinuity from the past trend due to changes in the real world that the model could not predict.

For example, 

It can be simplified as a first-order spline regression with \( M \) knots (changepoints):



\[ \tau(t) = b(t)^T \delta + kt + m \]

### Logistic Trend

The logistic trend in Pverse modifies the growth rate of the trend component over time:

\[ G = \frac{C}{1 + \exp(-\mathbf{B}\delta')} \]

### Seasonality

Pverse models seasonality using a Fourier series to approximate periodic functions:

\[ s(t) = \sum_{k=1}^K \left( a_k \cos\left(\frac{2\pi kt}{P}\right) + b_k \sin\left(\frac{2\pi kt}{P}\right) \right) \]

### Change Points

Change points in Pverse are handled using a Laplace prior on the rate adjustments at each changepoint. Users can define changepoint times or let them be uniformly distributed in the training data.

### Multivariate Models

Pverse supports multivariate forecasting, where the model components are estimated in the same way as for univariate models, but the likelihood is a multivariate distribution with a mean vector and covariance matrix.

## Advanced

### Generalized Linear Models (GLM)

Pverse extends the GAM concept by allowing users to define their own components and priors using the [Effects API](https://prophetverse.com/effects-api/). This API enables the creation of custom effects by subclassing [`BaseEffect`](https://prophetverse.com/reference/effects/).

### Linear Trend

In Pverse, the linear trend can be expressed in matrix form for computational efficiency:

\[ G = \mathbf{B}\delta' \]

where \(\mathbf{B}\) is the spline basis matrix and \(\delta'\) is a vector containing the rate adjustments, global rate, and global offset.

### Logistic Trend

The logistic trend in Pverse incorporates the logistic capacity \( C \) as a random variable:

\[ G = \frac{C}{1 + \exp(-\mathbf{B}\delta')} \]

### Seasonality

Seasonality in Pverse is modeled using a Fourier series, and the seasonal component can be efficiently computed using a design matrix:

\[ \mathbf{s} = \mathbf{X} \mathbf{\beta} \]

where \(\mathbf{X}\) is the Fourier basis matrix and \(\mathbf{\beta}\) contains the Fourier coefficients.

### Change Points

Change points in Pverse use a Laplace prior:

\[ \delta_i \sim \text{Laplace}(0, \sigma_{\delta}) \]

Changepoint times \(\kappa_i\) can be predefined or uniformly distributed. The offset and rate prior location are set based on the time series' extrema.

### Multivariate Models

In multivariate forecasting with Pverse, the likelihood is a multivariate normal distribution. The covariance matrix prior is modeled using an LKJ distribution.

### References

- For detailed explanations of GAMs and their applications, see [Hastie & Tibshirani (1990)](https://statweb.stanford.edu/~tibs/ElemStatLearn/)
- For a deeper understanding of Fourier series and their use in time series analysis, refer to [Percival & Walden (2000)](https://books.google.com/books/about/Wavelet_Methods_for_Time_Series_Analysis.html?id=v9nIoylk2gwC)
- For more on the LKJ distribution and its applications in Bayesian modeling, see [Lewandowski, Kurowicka, and Joe (2009)](https://projecteuclid.org/euclid.ba/1416240556)

