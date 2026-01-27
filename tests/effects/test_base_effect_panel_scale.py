import pandas as pd

from prophetverse.datasets._mmm.dataset1_panel import get_dataset
from prophetverse.effects.base import BaseEffect


class ExampleEffect(BaseEffect):
    """
    Example effect to validate scale broadcasting for panel data.
    """

    _tags = {
        "capability:panel": False,
        "capability:multivariate_input": True,
        "requires_X": True,
        "applies_to": "X",
        "filter_indexes_with_forecasting_horizon_at_transform": True,
        "requires_fit_before_transform": True,
    }

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        assert isinstance(scale, float)
        self.scale_received = scale


def test_panel_scale_is_broadcast_to_float():
    y, X, *_ = get_dataset()
    effect = ExampleEffect()

    scale = y.groupby(level=0).max()

    effect.fit(y=y, X=X, scale=scale)

    for eff in effect.effects_.values():
        assert isinstance(eff.scale_received, float)
