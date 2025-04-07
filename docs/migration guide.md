
## Overview

 This migration guide is intended for users familiar with `fbprophet` and outlines the key differences and necessary modifications for a seamless transition.

## Verifying Installation

Ensure that `prophetverse` is installed and accessible in your environment. To get started, follow the [installation guide](https://prophetverse.com/README.md/) for setting up Prophetverse.

```python
import prophetverse
print(prophetverse.__version__)
```

## Key Differences

### 1. Model Initialization

|Task|`fbprophet`|`prophetverse`|
|---|---|---|
|Import|`from prophet import Prophet`|`from prophetverse.sktime import Prophetverse`|
|Instantiate|`model = Prophet()`|`model = Prophetverse()`|

### 2. Future DataFrame Creation

Unlike `fbprophet`, `prophetverse` does not currently provide a built-in method for creating a future dataframe. This must be done manually:

```python
import pandas as pd

future_dates = pd.date_range(start=data["ds"].max(), periods=30, freq="D")
future = pd.DataFrame({"ds": future_dates})
```

### 3. Forecasting API

The fitting and prediction interfaces remain largely unchanged:

```python
model.fit(data)
forecast = model.predict(future)
```

### 4. Visualization

Plotting methods from `fbprophet` remain consistent:

```python
model.plot(forecast)
```

## Summary of API Changes

|Feature|`fbprophet`|`prophetverse`|
|---|---|---|
|Import path|`from prophet import Prophet`|`from prophetverse.sktime import Prophetverse`|
|Future dataframe|`model.make_future_dataframe()`|Manual via `pd.date_range()`|
|Prediction method|`model.predict()`|`model.predict()`|
|Visualization|`model.plot()`|`model.plot()`|


    
