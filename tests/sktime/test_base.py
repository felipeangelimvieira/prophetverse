import numpyro.distributions as dist
import pytest

from prophetverse.effects import LinearEffect
from prophetverse.effects.trend import PiecewiseLinearTrend
from prophetverse.engine import MAPInferenceEngine
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


def test_rshift_operator(base_effects_bayesian_forecaster):

    baseprophet = BaseProphetForecaster()
    trend = PiecewiseLinearTrend(
        changepoint_interval=10, changepoint_range=90, changepoint_prior_scale=1
    )

    # We need to create a custom Normal distribution for testing purposes
    class _Normal(dist.Normal):

        def __eq__(self, other):
            return isinstance(other, dist.Normal) and (
                self.loc == other.loc and self.scale == other.scale
            )

    effect = ("effect1", LinearEffect(prior=_Normal(10, 2)), r"(x1).*")
    effect_list = [("effect2", LinearEffect(prior=_Normal(10, 2)), r"(x1).*")]
    engine = MAPInferenceEngine()

    rshift_instance = baseprophet >> trend >> effect >> effect_list >> engine
    expected_instance = BaseProphetForecaster(
        trend=trend,
        exogenous_effects=[effect, *effect_list],
        inference_engine=engine,
    )

    assert rshift_instance == expected_instance


def test_samples_unset():
    model = BaseProphetForecaster()

    with pytest.raises(AttributeError):
        samples = model.posterior_samples_
