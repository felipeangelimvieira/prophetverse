from numpyro.infer import Predictive
from prophetverse.engine.base import BaseInferenceEngine
from numpyro import handlers
from jax import random


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

        _, trace_key, predictive_key = random.split(self._rng_key, 3)

        model = self.model_
        if self.substitute is not None:
            model = handlers.substitute(model, self.substitute)

        trace = handlers.trace(handlers.seed(model, trace_key)).get_trace(**kwargs)
        sample_sites = [
            site_name
            for site_name in trace.keys()
            if trace[site_name]["type"] == "sample"
            and not trace[site_name]["is_observed"]
        ]

        prior_predictive = Predictive(
            model,
            num_samples=self.num_samples,
            exclude_deterministic=True,
            return_sites=sample_sites,
        )
        self.posterior_samples_ = prior_predictive(predictive_key, **kwargs)

        if "obs" in self.posterior_samples_:
            # Remove the observed data from the samples
            del self.posterior_samples_["obs"]
        return self

    def _predict(self, **kwargs):
        """
        Draw samples from the prior predictive distribution.
        """
        _, predictive_key = random.split(self._rng_key)
        model = self.model_

        predictive = Predictive(
            model,
            posterior_samples=self.posterior_samples_,
            num_samples=self.num_samples,
        )

        self.samples_predictive_ = predictive(predictive_key, **kwargs)
        return self.samples_predictive_
        self.samples_predictive_ = predictive(predictive_key, **kwargs)
        return self.samples_predictive_
