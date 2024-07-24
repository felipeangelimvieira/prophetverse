import numpyro.distributions as dist
import pytest

from prophetverse.effects import LinearEffect
from prophetverse.sktime.base import BaseProphetForecaster


@pytest.fixture
def base_effects_bayesian_forecaster():
    return BaseProphetForecaster(
        exogenous_effects=[
            ("effect1", LinearEffect(prior=dist.Normal(10, 2)), r"(x1).*"),
        ]
    )


@pytest.mark.parametrize(
    "param_setting_dict",
    [
        dict(
            exogenous_effects=[
                ("effect1", LinearEffect(prior=dist.Laplace(0, 1)), r"(x1).*")
            ]
        ),
        dict(effect1__prior=dist.Laplace(0, 1)),
    ],
)
def test_set_params(base_effects_bayesian_forecaster, param_setting_dict):
    base_effects_bayesian_forecaster.set_params(**param_setting_dict)

    prior = base_effects_bayesian_forecaster.exogenous_effects[0][1].prior
    assert isinstance(prior, dist.Laplace)
    assert prior.loc == 0 and prior.scale == 1
    assert len(base_effects_bayesian_forecaster.exogenous_effects) == 1
    assert len(base_effects_bayesian_forecaster.exogenous_effects[0]) == 3
