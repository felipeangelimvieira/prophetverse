import jax.numpy as jnp
import pandas as pd
import pytest
import numpyro
import numpyro.distributions as dist
from prophetverse.effects.base import BaseAdditiveOrMultiplicativeEffect, BaseEffect


class ConcreteEffect(BaseAdditiveOrMultiplicativeEffect):
    """Most simple class to test abstracteffect methods."""

    _tags = {"requires_X": False}

    def _predict(self, data, predicted_effects, params) -> jnp.ndarray:
        """Calculate simple effect."""
        return jnp.mean(data, axis=1, keepdims=True)


@pytest.fixture(name="effect_with_regex")
def effect_with_regex():
    """Most simple class of abstracteffect with optional regex."""
    return ConcreteEffect()


@pytest.fixture
def effect_without_regex():
    """Most simple class of abstracteffect without optional regex."""
    return ConcreteEffect()


@pytest.mark.smoke
def test__predict(effect_with_regex):
    trend = jnp.array([1.0, 2.0, 3.0]).reshape((-1, 1))
    data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape((-1, 2))
    result = effect_with_regex.predict(data, predicted_effects={"trend": trend})
    expected_result = jnp.mean(data, axis=1).reshape((-1, 1))
    assert jnp.allclose(result, expected_result)

    call_result = effect_with_regex(data=data, predicted_effects={"trend": trend})

    assert jnp.all(call_result == result)


def test_bad_effect_mode():
    with pytest.raises(ValueError):
        BaseAdditiveOrMultiplicativeEffect(effect_mode="bad_mode")


def test_not_fitted():

    class EffectNotFitted(BaseEffect):
        _tags = {"requires_fit_before_transform": False}

    EffectNotFitted().transform(pd.DataFrame(), fh=pd.Index([]))

    class EffectMustFit(BaseEffect):
        _tags = {"requires_fit_before_transform": True}

    with pytest.raises(ValueError):
        EffectMustFit().transform(pd.DataFrame(), fh=pd.Index([]))


def test_broadcasting_columns():

    class SimpleEffect(BaseEffect):

        _tags = {
            "hierarchical_prophet_compliant": False,
            "capability:multivariate_input": False,
        }

        def _predict(self, data, predicted_effects, params):
            factor = numpyro.sample("factor", dist.Normal(0, 1))
            return data * factor

    effect = SimpleEffect()
    X = pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60], "exog2": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2021-01-01", periods=6),
    )
    Xt = effect.transform(X, fh=X.index)
    assert isinstance(Xt, list)
    assert len(Xt) == 2

    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=0):
        out = effect.predict(data=Xt)

    factor0 = trace["exog/factor"]["value"]
    factor1 = trace["exog2/factor"]["value"]

    assert factor0 != factor1
    expected = (X["exog"].values * factor0 + X["exog2"].values * factor1).reshape(
        (-1, 1)
    )
    assert jnp.allclose(out, expected), "Broadcasting effect prediction failed."


def test_broadcasting_panel():

    class SimpleEffect(BaseEffect):

        _tags = {
            "hierarchical_prophet_compliant": False,
            "capability:multivariate_input": False,
        }

        def _predict(self, data, predicted_effects, params):
            factor = numpyro.sample("factor", dist.Normal(0, 1))
            return data * factor

    effect = SimpleEffect()
    _X = pd.DataFrame(
        data={"exog": [10, 20, 30, 40, 50, 60], "exog2": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2021-01-01", periods=6),
    )
    X = {}
    for i, name in enumerate(["a", "b"]):
        X[i] = _X.copy()
        X[i] *= i + 1

    X = pd.concat(X, axis=0)

    Xt = effect.transform(X, fh=X.index.get_level_values(-1).unique())
    assert isinstance(Xt, list)
    assert len(Xt) == 2

    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=0):
        out = effect.predict(data=Xt)

    for i in X.index.get_level_values(0).unique():
        factor0 = trace[f"exog/panel-{i}/factor"]["value"]
        factor1 = trace[f"exog2/panel-{i}/factor"]["value"]

        assert factor0 != factor1
        expected = (
            X.loc[i, "exog"].values * factor0 + X.loc[i, "exog2"].values * factor1
        ).reshape((-1, 1))
        assert jnp.allclose(out[i], expected), "Broadcasting effect prediction failed."


def test_sample_params_warning():
    import warnings

    warnings.simplefilter("default", FutureWarning)
    with warnings.catch_warnings(record=True) as caught:

        class EffectWithSampleParams(BaseEffect):

            def _sample_params(self, data, predicted_effects):
                return {}

            def _predict(self, data, predicted_effects, params):
                return 0

    assert len(caught) == 1, "Expected exactly one warning"
    w = caught[0]
    assert issubclass(w.category, FutureWarning)


def test_update_data():

    effect = BaseEffect()

    # Array
    data_in = jnp.array([[1.0, 2.0]])
    data_out = jnp.array([[3.0, 4.0]])
    out = effect._update_data(data_in, data_out)
    assert jnp.array_equal(out, data_out), "Data update failed"

    # Tuple
    out = effect._update_data((data_in, None), data_out)
    assert jnp.array_equal(out[0], data_out), "Data update failed"
    assert out[1] is None, "Data update failed"

    # List
    data_in_list = [jnp.array([[1.0, 2.0]]), jnp.array([[3.0, 4.0]])]
    data_out_list = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    out = effect._update_data(data_in_list, data_out_list)
    assert len(out) == 2, "Data update failed"
    assert jnp.array_equal(out[0], data_out_list[:, [0]]), "Data update failed"
    assert jnp.array_equal(out[1], data_out_list[:, [1]]), "Data update failed"

    with pytest.raises(ValueError):
        effect._update_data("error", data_out)
