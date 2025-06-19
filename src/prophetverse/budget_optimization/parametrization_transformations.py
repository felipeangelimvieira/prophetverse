from prophetverse.budget_optimization.base import (
    BaseParametrizationTransformation,
)
import jax.numpy as jnp
from prophetverse.utils.frame_to_array import series_to_tensor

__all__ = [
    "IdentityTransform",
    "InvestmentPerChannelTransform",
    "TotalInvestmentTransform",
    "InvestmentPerChannelAndSeries",
    "InvestmentPerSeries",
]


class IdentityTransform(BaseParametrizationTransformation):
    """Identity decision variable transform.

    This transform does not change the decision variables. It is used when
    the decision variables are already in the correct format.
    """

    _tags = {
        "name": "identity",
        "backend": "scipy",
    }

    def _transform(self, x):
        return x

    def _inverse_transform(self, xt):
        return xt


class InvestmentPerChannelTransform(BaseParametrizationTransformation):
    """
    Change parametrization to be the share of each channel.

    Instead of parametrizing every time x channel investment, this transform
    parametrize to optimize per channel investment, while keeping the initial
    guess temporal share of the investment.
    """

    def _fit(self, X, horizon, columns):

        mask = X.index.get_level_values(-1).isin(horizon)
        X = X.loc[mask, columns]

        x_array = series_to_tensor(X)

        self.n_series_ = x_array.shape[0]
        # get daily share
        self.daily_share_ = x_array / x_array.sum(axis=(0, 1), keepdims=True)

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((self.n_series_, -1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x, axis=(0, 1)).flatten()
        # get the share of each column
        return x_sum

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share
        xt = xt.reshape((1, 1, len(self.columns_)))

        xt = xt * self.daily_share_
        xt = xt.flatten()
        return xt


class TotalInvestmentTransform(BaseParametrizationTransformation):
    """
    Change parametrization to be the total investment.

    Instead of parametrizing every time x channel investment, this transform
    parametrize to optimize per channel investment, while keeping the initial
    guess temporal share of the investment.
    """

    def _fit(self, X, horizon, columns):
        mask = X.index.get_level_values(-1).isin(horizon)
        X = X.loc[mask, columns]

        x_array = series_to_tensor(X)

        # get daily share
        self.daily_share_ = x_array / x_array.sum()

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((-1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x)
        # get the share of each column
        # Make sure x_sum is one dimensional
        x_sum = x_sum.reshape(-1)
        return x_sum

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share

        xt = xt * self.daily_share_
        xt = xt.flatten()
        return xt


class InvestmentPerChannelAndSeries(BaseParametrizationTransformation):

    def _fit(self, X, horizon, columns):

        mask = X.index.get_level_values(-1).isin(horizon)
        X = X.loc[mask, columns]

        x_array = series_to_tensor(X)

        self.n_series_ = x_array.shape[0]
        # get daily share
        self.daily_share_ = x_array / x_array.sum(axis=1, keepdims=True)

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((self.n_series_, -1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x, axis=1).flatten()

        return x_sum

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share
        xt = xt.reshape((self.n_series_, 1, len(self.columns_)))

        xt = xt * self.daily_share_
        xt = xt.flatten()
        return xt


class InvestmentPerSeries(BaseParametrizationTransformation):

    def _fit(self, X, horizon, columns):

        mask = X.index.get_level_values(-1).isin(horizon)
        X = X.loc[mask, columns]

        x_array = series_to_tensor(X)

        self.n_series_ = x_array.shape[0]
        # get daily share
        self.daily_share_ = x_array / x_array.sum(axis=(1, 2), keepdims=True)

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((self.n_series_, -1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x, axis=(1, 2)).flatten()

        return x_sum

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share
        xt = xt.reshape((self.n_series_, 1, 1))

        xt = xt * self.daily_share_
        xt = xt.flatten()
        return xt
