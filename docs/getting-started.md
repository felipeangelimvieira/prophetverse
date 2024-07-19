# Prophetverse


## Installation

To install with pip:

```bash
pip install prophetverse
```

Or with poetry:

```bash
poetry add prophetverse
```

## Forecasting with default values

The Prophetverse model provides an interface compatible with sktime.
Here's an example of how to use it:

```python
from prophetverse.sktime import Prophetverse

# Create the model
model = Prophetverse()

# Fit the model
model.fit(y=y, X=X)

# Forecast in sample
y_pred = model.predict(X=X, fh=y.index)
```
