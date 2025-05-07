import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from prophetverse.experimental.simulate import simulate
from prophetverse.sktime import HierarchicalProphet, Prophetverse


@pytest.mark.parametrize(
    "model,do",
    [
        (HierarchicalProphet(), {"exogenous_variables_effect/coefs": jnp.array([1])}),
        (Prophetverse(), {"exogenous_variables_effect/coefs": jnp.array([1])}),
    ],
)
def test_simulate(model, do):
    # NB: the new inference engine behaviour causes issues with convergence, so we fix by seeding for now, but
    # we might need to override r_hat in `simulate` if the inference engine has that property for a more sustainable fix
    np.random.seed(123)

    fh = pd.period_range(start="2022-01-01", periods=50, freq="M")
    X = pd.DataFrame(index=fh, data={"x1": list(range(len(fh)))})
    samples, model = simulate(model=model, fh=fh, X=X, do=do, return_model=True)
    assert isinstance(samples, pd.DataFrame)
    assert (
        samples.index.get_level_values(0).nunique()
        == model.inference_engine_.num_samples
    )
    assert samples.index.get_level_values(-1).nunique() == len(fh)

    expected_intervention = jnp.arange(len(fh))
    assert jnp.all(
        samples["exogenous_variables_effect"].values
        == jnp.tile(
            expected_intervention,
            (model.inference_engine_.num_samples,),
        ).flatten()
    )
