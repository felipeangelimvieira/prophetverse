from numpyro.infer import Predictive
from prophetverse.engine.base import BaseInferenceEngine
from numpyro import handlers


class PriorPredictiveInferenceEngine(BaseInferenceEngine):
    """
    Prior‐only inference engine for prior predictive checks.
    Samples parameters from their priors and runs the model.
    """

    _tags = {"inference_method": "prior_predictive"}

    def __init__(self, num_samples=1000, rng_key=None, substitute=None):
        self.num_samples = num_samples
        self.substitute = substitute
        super().__init__(rng_key)

    def _infer(self, **kwargs):

        model = self.model_
        if self.substitute is not None:
            model = handlers.substitute(model, self.substitute)

        prior_predictive = Predictive(
            model,
            num_samples=self.num_samples,
            exclude_deterministic=True,
        )
        self.posterior_samples_ = prior_predictive(self._rng_key, **kwargs)

        if "obs" in self.posterior_samples_:
            # Remove the observed data from the samples
            del self.posterior_samples_["obs"]
        return self

    def _predict(self, **kwargs):
        """
        Draw samples from the prior predictive distribution.
        """
        model = self.model_

        predictive = Predictive(
            model,
            posterior_samples=self.posterior_samples_,
            num_samples=self.num_samples,
        )

        self.samples_predictive_ = predictive(self._rng_key, **kwargs)
        return self.samples_predictive_
