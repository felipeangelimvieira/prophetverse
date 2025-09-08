from prophetverse.utils.plotting import plot_prior_predictive
from prophetverse import LinearEffect
import pandas as pd
import jax.numpy as jnp


def test_plot_prior_predictive():
    """Test the plot_prior_predictive function."""
    instance = LinearEffect()
    X = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5]}, index=pd.date_range("2023-01-01", periods=5)
    )
    y = X * 2
    fig, ax = plot_prior_predictive(
        instance,
        X=X,
        y=y,
        predicted_effects={"trend": jnp.array([[2], [4], [6], [8], [10]])},
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) > 0
    assert len(ax.collections) > 0
