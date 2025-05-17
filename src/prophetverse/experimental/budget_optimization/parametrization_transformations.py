from prophetverse.experimental.budget_optimization.base import (
    BaseParametrizationTransformation,
)
import jax.numpy as jnp


__all__ = [
    "IdentityTransform",
    "InvestmentPerChannelTransform",
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
        X = X.loc[horizon, columns]

        x_array = jnp.array(X.values)

        # get daily share
        self.daily_share_ = x_array / x_array.sum(axis=0)

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((-1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x, axis=0)
        # get the share of each column
        return x_sum

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share
        xt = xt.reshape((-1, len(self.columns_)))

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
        X = X.loc[horizon, columns]

        x_array = jnp.array(X.values)

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
