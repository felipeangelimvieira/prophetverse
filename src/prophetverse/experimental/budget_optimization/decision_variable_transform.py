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


class OptimizeChannelShare(BaseDecisionVariableTransform):

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


class OptimizeObservationShare(BaseDecisionVariableTransform):
    """Optimize data share.

    This transform is used to optimize the share of data used for
    forecasting. It is used when the decision variables are in the form of
    data share.
    """

    _tags = {
        "name": "optimize_data_share",
        "backend": "scipy",
    }

    def _fit(self, X, horizon, columns):
        X = X.loc[horizon, columns]

        x_array = jnp.array(X.values)

        # get daily share
        self.total_budget_ = x_array.sum()

    def _transform(self, x):
        x = x / x.sum()
        return x

    def _inverse_transform(self, xt):
        xt = jnp.exp(xt) / jnp.exp(xt).sum()
        xt = xt * self.total_budget_
        return xt
