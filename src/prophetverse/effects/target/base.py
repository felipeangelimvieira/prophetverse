from prophetverse.effects.base import BaseEffect


class BaseTargetEffect(BaseEffect):
    """Base class for effects."""

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "capability:panel": False,
        # If no columns are found, should
        # _predict be skipped?
        "skip_predict_if_no_match": False,
        # Should only the indexes related to the forecasting horizon be passed to
        # _transform?
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def _transform(self, X, fh):
        if X is not None:
            return super()._transform(X, fh)
        return None
