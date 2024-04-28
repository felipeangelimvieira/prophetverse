# Hierarchical Prophet
<p align="center">
<img src="docs/static/logo.webp" width="200">

</p>

[![PyPI version](https://badge.fury.io/py/prophetverse.svg)](https://badge.fury.io/py/prophetverse)

This library was created to make a numpyro-based Prophet model for timeseries forecasting. In addition, it also provides an extension that allows for hierarchical forecasting simultaneously, with potentially shared coefficients between timeseries. All implementations (hierarchical and univariate) are based on sktime interface.

The hierarchical one creates a Prophet-like model for each bottom series, but uses a multivariate normal likelihood as likelihood. Checkout the docs for more example!


## Installation

To install with pip:

```bash
pip install prophetverse
```

Or with poetry:

```bash
poetry add prophetverse
```

