from prophetverse.experimental.budget_optimization.base import (
    BaseDecisionVariableTransform,
)
import jax.numpy as jnp


class IdentityDecisionVariableTransform(BaseDecisionVariableTransform):
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


class OptimizeInvestmentShare(BaseDecisionVariableTransform):

    def _fit(self, X, horizon, columns):
        X = X.loc[horizon, columns]

        x_array = jnp.array(X.values)

        # get daily share
        self.daily_share_ = x_array / x_array.sum(axis=1)[:, None]

    def _transform(self, x: jnp.ndarray):
        # get share per column
        # x is a (N*M) array
        x = x.reshape((-1, len(self.columns_)))
        # get the sum of each row
        x_sum = jnp.sum(x, axis=1)
        # get the share of each column
        x_share = x / x_sum[:, None]
        x_share = x_share.flatten()
        return x_share

    def _inverse_transform(self, xt):
        # Multiply each column share by the daily share
        xt = xt.reshape((-1, len(self.columns_)))
        xt = xt * self.daily_share_
        xt = xt.flatten()
        return xt
