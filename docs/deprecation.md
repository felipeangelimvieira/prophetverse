# Prophetverse


__Prophetverse__ leverages the __Generalized Additive Model (GAM)__ idea in the original
Prophet model and extends it to be more flexible and customizable. 
Roughly, GAMs are a type of 
statistical model used for regression analysis that models the expected value of 
the endogenous variable $Y$ as the sum of many functions $\{f_i\}_{i=1}^n$ of exogenous
variables $\{x_i\}_{i=1}^n$. 

$$
\mathbb{E}[Y] = f_1(x_1) + f_2(x_2) + \ldots + f_n(t)\text{, }\quad n \in \mathbb{N}
$$


The key idea of Prophet is using bayesian GAMs to model timeseries, and, 
instead of trying to approximate the timeseries through some kind of auto-regressive
model, treat it as a curve fitting exercise. This leads to fast, interpretable and 
accurate forecasting.

Before digging into Prophetverse, let's review the original Prophet model and its
components.

## Meta's Prophet

The bayesian model of Prophet uses a normal likelihood to model observations, putting
a HalfCauchy hyperprior into the standard deviation:

$$
y \sim \mathcal{N}(\hat{y}_{mean}, \sigma^2)\quad \text{where} \quad
\sigma \sim HalfCauchy(0.5)
$$

The mean term $\hat{y}_{mean}$ is composed of several components: trend, seasonality,
holidays, and other regressors. The components can be additive and multiplicative. 
The additive can be written as:



$$
\hat{y}_{mean}(t) = \hat{\tau}(t) + \hat{s}(t) + \hat{h}(t) + \hat{r}(t)
$$

where $\hat{\tau}(t)$ is the trend component, $\hat{s}(t)$ is the seasonality component,
$\hat{h}(t)$ is the holiday component, and $\hat{v}(t)$ is other regressors components.

The multiplicative version scales each component by the trend at time $t$:

$$
\hat{y}_{mean}(t) = \hat{\tau}(t)  + \hat{\tau}(t)\cdot\hat{s}(t) +
 \hat{\tau}(t)\cdot\hat{h}(t) + \hat{\tau}(t)\cdot\hat{v}(t)
$$

Although we separate seasonality, holidays and exogenous variables in these decompositions,
they are essentially estimated in the same way: as a linear combination of exogenous
features. In the case of seasonality, the exogenous features are fourier terms; for
holidays, dummy variables; and regressors are defined by the user.

!!! note
    One way to interpret the differences between the additive and multiplicative versions is
    thinking about the unit of each component. For example, if we are forecasting the amount
    of money (USD) a company makes in a day, the additive component (e.g., seasonality $s(t)$)
    will be forecasted in USD, while the multiplicative component will be forecasted as a
    percentage. 

### Trend

There are mainly two types of trends supported: linear and logistic. We will first
take a look at the original mathematical formulation of Prophet's paper, and then
simplify it to obtain a simpler and more interpretable version. In that sense, don't
dedicate too much time to understand the original formulation, as it is expected that
the simplified version will help you the most.

#### Linear trend

The linear trend is modeled as a piecewise linear functions with changepoints. Let $M$ be
the number of changepoints, $\delta \in \mathbb{R}^M$ be the rate adjustment at each changepoint, $\{\kappa_i\}_{i=1}^M$
be the changepoint times, and be $a(t) \in \{0,1\}^M$ be a vector which assumes 1 if the
corresponding changepoint is greater than $t$ and 0 otherwise. In addition, let $k$ represent
the global rate and $m$ the global offset. Then, the linear trend
is defined as:

$$
\tau(t) = (k + a(t)^T\delta)t + (m + a(t)^T\gamma), \quad \text{where} \quad \gamma_i = \kappa_i\delta_i
$$

The first part accounts for the rate adjustment at each changepoint, and the second part
corrects the offset at each changepoint, so that the trend is continuous. This can be
simplified as a first-order spline regression with $M$ knots (changepoints). Let 
$b(t) \in \mathbb{R}^M$ be a vector so that $b(t)_i = (t - \kappa_i)_+$ (the positive part of
$t - \kappa_i$). Then, the linear trend can be written as:

$$
\tau(t) = (b(t)^T \delta) \, t + kt + m
$$

We can also write the trend for all timestamps as a matrix multiplication. Let
$\mathbf{B} \in \mathbb{R}^{T \times M+2}$ be the matrix whose rows are $b'(t) = \left[ b(t), t, 1 \right]$ 
for each time $t$. In other words, it is the spline basis matrix. The $t$ and $1$ at the end of the vector are included to account for the
global rate and offset. Furthermore, consider the vector $\delta' = \left[ \delta, k, m \right]$.
Then, the trend vector $G \in \mathbb{R}^T$, $G_i = \tau(\mathbf{t}_i)$, can be written as:

\begin{align}
G &= \mathbf{B}\delta' \\
\end{align}

\begin{align}
G &=  \begin{bmatrix}
(t_0 - \kappa_0)_+ & (t_0 - \kappa_1)_+ & \ldots & (t_0 - \kappa_{M-1})_+ & t_0 & 1 \\
(t_1 - \kappa_0)_+ & (t_1 - \kappa_1)_+ & \ldots & (t_1 - \kappa_{M-1})_+ & t_1 & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
(t_{T-1} - \kappa_0)_+ & (t_{T-1} - \kappa_1)_+ & \ldots & (t_{T-1} - \kappa_{M-1})_+ & t_{T-1} & 1 \\
\end{bmatrix} \begin{bmatrix}
\delta_0 \\
\delta_1 \\
\vdots \\
\delta_{M-1} \\
k \\
m \\
\end{bmatrix}
\end{align}


!!! Example

    One possible realization of $\mathbf{B}$ is:

    $$
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 1 & 1 \\
    1 & 0 & 0 & 0 & 2 & 1 \\
    2 & 1 & 0 & 0 & 3 & 1 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
    T-1 & T-2 & T-3 & T-4 & T-1 & 1 \\
    \end{bmatrix}
    $$


#### Logistic trend

The logistic trend of the original model uses the piecewise logistic linear trend to 
change the rate at which
$t$ grows. We will not explain the mathematical formulation of the original paper here,
and will already leverage what we have learned about the linear trend to simplify it.

$$
G = \frac{C}{1 + \exp(-\mathbf{B}\delta')}
$$

where $C$ is the logistic capacity, which should be passed as input to the model (in the 
original Prophet, not in Prophetverse).

#### Changepoint priors

The original model puts a Laplace prior on the rate adjustment $\delta_i \sim Laplace(0, \sigma_{\delta})$
where $\sigma_{\delta}$ is a hyperparameter. The changepoint times $\kappa_i$ can be 
predefined by the user, or can be uniformly distributed in the training data.


### Seasonality

To model seasonality, Prophet uses a Fourier series to approximate periodic functions, allowing the model to fit complex seasonal patterns flexibly. This approach involves determining the number of Fourier terms (`K`), which corresponds to the complexity of the seasonality. The formula for a seasonal component `s(t)` in terms of a Fourier series is given as:

$$
s(t) = \sum_{k=1}^K \left( a_k \cos\left(\frac{2\pi kt}{P}\right) + b_k \sin\left(\frac{2\pi kt}{P}\right) \right)
$$

Here, `P` is the period (e.g., 365.25 for yearly seasonality), and $a_k$ and $b_k$ are the Fourier coefficients that the model estimates. The choice of `K` depends on the granularity of the seasonal changes one wishes to capture.
A Normal prior is placed on the coefficients, $a_k, b_k \sim \mathcal{N}(0, \sigma_s)$, where $\sigma_s$ is a hyperparameter.


#### Matrix Formulation of Fourier Series

To efficiently compute the seasonality for multiple time points, we can represent the Fourier series in a matrix form. This method is especially useful for handling large datasets and simplifies the implementation of the model in computational software.

**Design Matrix Construction**:

Let `T` be the number of time points, and create a design matrix `X` of size `T x 2K`. Each row of `X` corresponds to a time point and contains all Fourier basis functions evaluated at that time:

$$
\mathbf{X} = \begin{bmatrix}
\cos\left(\frac{2\pi \cdot 1 \cdot t_1}{P}\right) & \sin\left(\frac{2\pi \cdot 1 \cdot t_1}{P}\right) & \cdots & \cos\left(\frac{2\pi \cdot K \cdot t_1}{P}\right) & \sin\left(\frac{2\pi \cdot K \cdot t_1}{P}\right) \\
\cos\left(\frac{2\pi \cdot 1 \cdot t_2}{P}\right) & \sin\left(\frac{2\pi \cdot 1 \cdot t_2}{P}\right) & \cdots & \cos\left(\frac{2\pi \cdot K \cdot t_2}{P}\right) & \sin\left(\frac{2\pi \cdot K \cdot t_2}{P}\right) \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\cos\left(\frac{2\pi \cdot 1 \cdot t_T}{P}\right) & \sin\left(\frac{2\pi \cdot 1 \cdot t_T}{P}\right) & \cdots & \cos\left(\frac{2\pi \cdot K \cdot t_T}{P}\right) & \sin\left(\frac{2\pi \cdot K \cdot t_T}{P}\right)
\end{bmatrix}
$$

**Coefficient Vector**:

Define a vector $\beta$ of length `2K` containing the coefficients $a_1, b_1, \dots, a_K, b_K$:

$$
\mathbf{\beta} = \begin{bmatrix}
a_1 \\
b_1 \\
\vdots \\
a_K \\
b_K
\end{bmatrix}
$$

The seasonality for all time points can then be computed through the matrix product of $X$ and $\beta$:

$$
\mathbf{s} = \mathbf{X} \mathbf{\beta}
$$

Each element of vector $s$, denoted as $s_i$, represents the seasonality at time $t_i$.

This matrix approach not only makes the computation faster and more scalable but also simplifies integration with other components of the forecasting model.




### Holidays and others

All holidays and other regressors are modeled in the same way: as a linear combination
of exogenous features. The holiday dates are converted to a binary matrix, where each
column represents a holiday and each row a date. The value of the matrix is 1 if the
corresponding date is a holiday and 0 otherwise. The matrix is then multiplied by a
vector of coefficients, which are estimated by the model. The same is done for other
regressors, which are defined by the user.


## Prophetverse

<p align="center">
<img src="/static/prophetverse-universe.png">

</p>

### Univariate

Prophetverse extends this idea in flexible API that allows users to define their own
components and priors. In addition to the Normal Likelihood, we have two other likelihood functions for observations
that can be extremely useful: Gamma and Negative Binomial. 

$$
y \sim \mathcal{Likelihood}(\phi(y_{mean}), \sigma^2)\quad \text{where} \quad
\sigma \sim HalfNormal(\sigma_{hyper})
$$

$$
y_{mean}(t) = \tau(t)  + \sum_{i=1}^n f_i(\tau(t), x_i(t))
$$

Where $\phi$ is a function that maps the mean to the support of the likelihood. For normal
likelihood, $\phi$ is the identity function, but for Gamma and Negative Binomial, it is

$$
\phi(k) = \begin{cases}
k & \text{if } k > 10^5 \\
z\exp(k-z) & \text{if } k \leq z
\end{cases}
$$

for some small threshold $z$. We set $z = 10^{-5}$ in our implementation. The reason for this
is to avoid zero or negative values in the support of the likelihood, which can lead to
error.

The functions $f_i$ can be defined in code through [Effects API](reference/effects),
and can be any function that maps trend and an exogenous variable $x_i(t)$ to a
real number. The trend is ignored if a particular component is additive. 
The trend can also be customized, but the `linear` and `logistic` are readly
available. 

One main difference between Prophet and Prophetverse logistic trend is that in the
latter, the capacity is modeled as a random variable and is assumed to be constant. In the original model, it was a an input variable and this necessary to pass the capacity as a hyperparameter, but we often don't know it.


### Multivariate

Prophetverse also supports multivariate forecasting. In this case, the model is
essentially the same, but for now only Normal Likelihood is supported. Depending on
the usage of the library, we may add other likelihoods in the future (please open an
issue if you need it!). In that case, all other components are estimated in the same way,
but the likelihood is a multivariate distribution. The mean of the distribution is
a vector, and the covariance matrix prior is a LKJ distribution.