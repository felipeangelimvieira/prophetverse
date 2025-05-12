from numpyro.infer import Predictive
from prophetverse.engine.base import BaseInferenceEngine
from numpyro import handlers


class PriorPredictiveInferenceEngine(BaseInferenceEngine):
    """
    Prior‐only inference engine for prior predictive checks.
    Samples parameters from their priors and runs the model.
    """

    _tags = {"inference_method": "prior_predictive"}

    def __init__(self, num_samples=1000, rng_key=None):
        self.num_samples = num_samples
        super().__init__(rng_key)

    def _infer(self, **kwargs):
        # sample parameters from prior via numpyro.handlers.seed and trace

        prior_predictive = Predictive(
            self.model_, num_samples=self.num_samples, exclude_deterministic=False
        )
        self.posterior_samples_ = prior_predictive(self._rng_key, **kwargs)
        del self.posterior_samples_["obs"]
        return self

    def _predict(self, **kwargs):
        """
        Draw samples from the prior predictive distribution.
        """
        predictive = Predictive(
            self.model_, self.posterior_samples_, num_samples=self.num_samples
        )

        self.samples_predictive_ = predictive(self._rng_key, **kwargs)
        return self.samples_predictive_
