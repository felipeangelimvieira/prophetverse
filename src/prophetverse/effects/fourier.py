"""Fourier effects for time series forecasting with seasonality."""

from typing import Dict, List, Union, Optional

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.effects.base import EFFECT_APPLICATION_TYPE, BaseEffect
from prophetverse.effects.linear import LinearEffect
from prophetverse.sktime._expand_column_per_level import ExpandColumnPerLevel

__all__ = ["LinearFourierSeasonality"]


def _coerce_period(value, index: pd.Index):
    """Coerce *value* to a type that is comparable with *index*.

    Works for both :class:`pandas.DatetimeIndex` and
    :class:`pandas.PeriodIndex`.

    Parameters
    ----------
    value : str, pd.Timestamp, pd.Period, or None
        The boundary value supplied by the user.
    index : pd.Index
        The time index whose dtype drives the coercion.

    Returns
    -------
    pd.Timestamp or pd.Period
    """
    if value is None:
        return None
    if isinstance(index, pd.PeriodIndex):
        # Ensure any Period input is coerced to the index frequency so that
        # comparisons like `index >= value` do not raise IncompatibleFrequency.
        if isinstance(value, pd.Period):
            if value.freq == index.freq:
                return value
            try:
                return value.asfreq(index.freq)
            except (ValueError, TypeError):
                # Fallback: construct a new Period from the timestamp representation.
                return pd.Period(value.to_timestamp(), freq=index.freq)
        return pd.Period(value, freq=index.freq)
    # DatetimeIndex (or anything else – fall back to Timestamp)
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Period):
        return value.to_timestamp()
    return pd.Timestamp(value)


class LinearFourierSeasonality(BaseEffect):
    """Linear Fourier Seasonality effect.

    Compute the linear seasonality using Fourier features.

    Optionally, "start_period" and "end_period" can be used to restrict
    the seasonality to a sub-range of the time axis.  Time-steps outside
    "[start_period, end_period]" are forced to zero.

    Parameters
    ----------
    sp_list : List[float]
        List of seasonal periods.
    fourier_terms_list : List[int]
        List of number of Fourier terms to use for each seasonal period.
    freq : str
        Frequency of the time series. Example: "D" for daily, "W" for weekly, etc.
    prior_scale : float, optional
        Scale of the prior distribution for the effect, by default 1.0.
    effect_mode : str, optional
        Either "multiplicative" or "additive" by default "additive".
    linear_effect : LinearEffect, optional
        Custom LinearEffect instance used internally.  When *None* a
        default LinearEffect(prior=Normal(0, prior_scale)) is created.
    start_period : str, pd.Timestamp, or pd.Period, optional
        Start of the active window for this seasonality component.  Time
        steps before this date/period are multiplied by zero.  When
        None (default) the component is active from the very first
        observation.
    end_period : str, pd.Timestamp, or pd.Period, optional
        End of the active window for this seasonality component.  Time
        steps after this date/period are multiplied by zero.  When
        None (default) the component is active until the very last
        observation.
    """

    _tags = {
        # Supports multivariate data? Can this
        # Effect be used with Multiariate prophet?
        "hierarchical_prophet_compliant": True,
        # If no columns are found, should
        # _predict be skipped?
        "requires_X": False,
        "capability:panel": True,
    }

    def __init__(
        self,
        sp_list: List[float],
        fourier_terms_list: List[int],
        freq: Union[str, None],
        prior_scale: float = 1.0,
        effect_mode: EFFECT_APPLICATION_TYPE = "additive",
        linear_effect: Optional[LinearEffect] = None,
        start_period=None,
        end_period=None,
    ):
        self.sp_list = sp_list
        self.fourier_terms_list = fourier_terms_list
        self.freq = freq
        self.prior_scale = prior_scale
        self.effect_mode = effect_mode
        self.linear_effect = linear_effect
        self.start_period = start_period
        self.end_period = end_period

        super().__init__()

        self.expand_column_per_level_ = None  # type: Union[None,ExpandColumnPerLevel]

        self._linear_effect = (
            linear_effect
            if linear_effect is not None
            else LinearEffect(
                prior=dist.Normal(0, self.prior_scale), effect_mode=self.effect_mode
            )
        )

        self.set_tags(
            **{
                "capability:panel": self._linear_effect.get_tag("capability:panel"),
                "capability:multivariate_input": self._linear_effect.get_tag(
                    "capability:multivariate_input"
                ),
            }
        )

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0):
        """Customize the initialization of the effect.

        Fit the fourier feature transformer and the linear effect.

        Parameters
        ----------
        y : pd.DataFrame
            The timeseries dataframe

        X : pd.DataFrame
            The DataFrame to initialize the effect.

        scale: float, optional
            The scale of the timeseries, by default 1.0.
        """
        self.fourier_features_ = FourierFeatures(
            sp_list=self.sp_list,
            fourier_terms_list=self.fourier_terms_list,
            freq=self.freq,
            keep_original_columns=False,
        )

        self.fourier_features_.fit(X=X)
        X = self.fourier_features_.transform(X)

        is_panel = X.index.nlevels > 1 and X.index.droplevel(-1).nunique() > 1
        if is_panel and not self.get_tag("capability:panel", False):
            self.expand_column_per_level_ = ExpandColumnPerLevel([".*"]).fit(X=X)
            X = self.expand_column_per_level_.transform(X)  # type: ignore

        self.linear_effect_ = self._linear_effect.clone()

        self.linear_effect_.fit(X=X, y=y, scale=scale)

    def _build_mask(self, X: pd.DataFrame) -> Optional[jnp.ndarray]:
        """Build a binary time-range mask from "start_period" to "end_period".

        Returns a float array of shape (T,) where T is the number of
        unique time-steps in X, or None when neither boundary is set.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame whose index contains the time dimension (possibly a
            MultiIndex whose last level is time).

        Returns
        -------
        jnp.ndarray of shape (T,) or None
        """
        if self.start_period is None and self.end_period is None:
            return None

        time_index = X.index.get_level_values(-1).unique()

        start = _coerce_period(self.start_period, time_index)
        end = _coerce_period(self.end_period, time_index)

        if start is not None and end is not None and start > end:
            raise ValueError(
                f"start_period ({self.start_period!r}) must be earlier than or "
                f"equal to end_period ({self.end_period!r})."
            )

        mask = np.ones(len(time_index), dtype=np.float32)
        if start is not None:
            mask *= (time_index >= start).astype(np.float32)
        if end is not None:
            mask *= (time_index <= end).astype(np.float32)

        return jnp.array(mask)

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> dict:
        """Prepare input data to be passed to numpyro model.

        This method return a jnp.ndarray of sines and cosines of the given
        frequencies, together with an optional binary mask that forces the
        effect to zero outside start_period, end_period.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the exogenous variables for the training
            time indexes, if passed during fit, or for the forecasting time indexes, if
            passed during predict.

        fh : pd.Index
            The forecasting horizon as a pandas Index.

        Returns
        -------
        dict
            Dictionary with key "data" (the Fourier feature array) and,
            when a time-range restriction is active, key "mask" (shape(T,) float array).
        """

        mask = self._build_mask(X)

        X = self.fourier_features_.transform(X)

        if self.expand_column_per_level_ is not None:
            X = self.expand_column_per_level_.transform(X)

        array = self.linear_effect_.transform(X, fh)

        out = {"data": array}
        if mask is not None:
            out["mask"] = mask
        return out

    def _predict(
        self, data: Dict, predicted_effects: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """Apply and return the effect values.

        Parameters
        ----------
        data : Any
            Data obtained from the transformed method.

        predicted_effects : Dict[str, jnp.ndarray], optional
            A dictionary containing the predicted effects, by default None.

        Returns
        -------
        jnp.ndarray
            An array with shape (T,1) for univariate timeseries, or (N, T, 1) for
            multivariate timeseries, where T is the number of timepoints and N is the
            number of series.
        """
        result = self.linear_effect_.predict(
            data=data["data"],
            predicted_effects=predicted_effects,
        )

        mask = data.get("mask", None)
        if mask is not None:
            # result: (T, 1)  →  mask reshape: (T, 1)
            # result: (N, T, 1) →  mask reshape: (1, T, 1)
            if result.ndim == 2:
                mask = mask.reshape(-1, 1)
            else:
                mask = mask.reshape(1, -1, 1)
            result = result * mask

        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {
                "sp_list": [7],
                "fourier_terms_list": [1],
                "freq": "D",
                "prior_scale": 1.0,
                "effect_mode": "additive",
            },
            {
                "sp_list": [7],
                "fourier_terms_list": [1],
                "freq": "D",
                "prior_scale": 1.0,
                "effect_mode": "additive",
                "start_period": "2021-01-04",
                "end_period": "2021-01-08",
            },
        ]
